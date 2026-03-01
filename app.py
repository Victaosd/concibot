import base64
import io
import json
import re
from typing import List, Dict, Optional

import pandas as pd
import pdfplumber
import plotly.express as px
import streamlit as st
from openai import OpenAI
from pdf2image import convert_from_bytes
from PIL import Image


# -----------------------------
# Configuração básica da página
# -----------------------------
st.set_page_config(
    page_title="ConciBot - Classificador de Extratos",
    page_icon="🧾",
    layout="wide",
)


# ---------------------------------------
# Constantes e textos de sistema para a IA
# ---------------------------------------
SYSTEM_PROMPT = (
    "Você é um assistente contábil brasileiro especializado em classificação de "
    "lançamentos bancários. Para cada lançamento, classifique em uma das categorias: "
    "Receita, Fornecedor, Impostos e Taxas, Folha de Pagamento, Despesa Operacional, "
    "Transferência, ou Revisar (quando não tiver certeza). Retorne apenas um JSON "
    "com array de objetos contendo: data, descricao, valor, categoria, confianca "
    "(alta/media/baixa)."
)

EXTRACTION_SYSTEM_PROMPT = (
    "Você é um assistente contábil brasileiro especializado em extrair "
    "lançamentos bancários de textos de extratos de bancos brasileiros "
    "(Nubank, Itaú, Bradesco, Santander, Banco do Brasil, Caixa, etc.). "
    "Sua tarefa é identificar todos os lançamentos contendo: data, descrição "
    "e valor (positivo para créditos, negativo para débitos). "
    "Retorne apenas um JSON com o campo 'lancamentos', que é um array de objetos "
    "no formato: {data, descricao, valor}. A data deve estar em um formato "
    "reconhecível (por exemplo, dd/mm/aaaa) e o valor deve ser um número decimal "
    "com ponto ou vírgula como separador decimal."
)


# ----------------------------------------
# Funções auxiliares de parsing e limpeza
# ----------------------------------------
def parse_brazilian_number(value) -> Optional[float]:
    """Converte um número em formato brasileiro (R$ 1.234,56) para float."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()
    s = s.replace("R$", "").replace(" ", "")
    # Remove separador de milhar e troca vírgula decimal por ponto
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def normalize_date(value) -> Optional[str]:
    """Converte datas em vários formatos para uma string padronizada AAAA-MM-DD."""
    try:
        dt = pd.to_datetime(value, dayfirst=True, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.date().isoformat()
    except Exception:
        return None


def try_detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """Tenta identificar automaticamente a coluna de data em um DataFrame genérico."""
    date_keywords = [
        "data",
        "date",
        "dt",
        "transaction date",
        "posting date",
        "fecha",
        "fechamento",
    ]
    best_col = None
    best_score = 0.0

    for col in df.columns:
        series = df[col].dropna().astype(str)
        if series.empty:
            continue
        sample = series.head(100)
        valid_count = sample.apply(lambda v: normalize_date(v) is not None).sum()
        ratio = valid_count / len(sample)

        name = str(col).lower()
        if any(k in name for k in date_keywords):
            ratio += 0.2  # pequeno bônus por palavra-chave no nome

        if ratio > best_score and ratio >= 0.5:
            best_score = ratio
            best_col = col

    return best_col


def try_detect_value_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Tenta identificar automaticamente coluna(s) de valor.

    Retorna dicionário com chaves:
      - "single": nome de coluna única com o valor, OU
      - "debit" e "credit": nomes de colunas separadas (débito/crédito)
    """
    value_keywords = ["valor", "value", "amount", "quantia", "total"]
    debit_keywords = ["deb", "déb", "saída", "saida", "pago", "pagamento"]
    credit_keywords = ["cred", "créd", "entrada", "receb", "receita"]

    numeric_candidates: List[Dict] = []

    for col in df.columns:
        series = df[col]
        sample = series.dropna().astype(str).head(100)
        if sample.empty:
            continue
        parsed = sample.apply(lambda v: parse_brazilian_number(v) is not None).sum()
        ratio = parsed / len(sample)
        if ratio < 0.6:
            continue
        name = str(col).lower()
        score = ratio
        if any(k in name for k in value_keywords):
            score += 0.2
        numeric_candidates.append({"col": col, "score": score, "name": name})

    if not numeric_candidates:
        return {"single": None, "debit": None, "credit": None}

    # Ordena por score decrescente
    numeric_candidates.sort(key=lambda x: x["score"], reverse=True)

    # Caso simples: apenas uma coluna boa de valor
    if len(numeric_candidates) == 1:
        return {
            "single": numeric_candidates[0]["col"],
            "debit": None,
            "credit": None,
        }

    # Caso: possíveis colunas de débito e crédito
    debit_col = None
    credit_col = None
    for cand in numeric_candidates[:3]:
        if debit_col is None and any(k in cand["name"] for k in debit_keywords):
            debit_col = cand["col"]
        if credit_col is None and any(k in cand["name"] for k in credit_keywords):
            credit_col = cand["col"]

    if debit_col or credit_col:
        return {"single": None, "debit": debit_col, "credit": credit_col}

    # fallback: pega a melhor coluna como valor único
    return {
        "single": numeric_candidates[0]["col"],
        "debit": None,
        "credit": None,
    }


def try_detect_description_column(
    df: pd.DataFrame, exclude: List[str]
) -> Optional[str]:
    """Tenta identificar automaticamente a coluna de descrição (texto)."""
    desc_keywords = [
        "descri",
        "hist",
        "lança",
        "histor",
        "memo",
        "description",
        "detalhe",
        "details",
    ]
    best_col = None
    best_score = 0.0

    for col in df.columns:
        if col in exclude:
            continue
        series = df[col]
        sample = series.dropna().astype(str).head(100)
        if sample.empty:
            continue
        # Queremos colunas predominantemente textuais
        numeric_like = sample.apply(
            lambda v: parse_brazilian_number(v) is not None
        ).sum()
        numeric_ratio = numeric_like / len(sample)
        if numeric_ratio > 0.3:
            continue

        name = str(col).lower()
        score = 1.0 - numeric_ratio
        if any(k in name for k in desc_keywords):
            score += 0.2

        if score > best_score:
            best_score = score
            best_col = col

    return best_col


def normalize_statement_dataframe(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    A partir de um DataFrame genérico, tenta produzir um DataFrame padronizado
    com colunas: data, descricao, valor.
    """
    if df is None or df.empty:
        return None

    date_col = try_detect_date_column(df)
    value_info = try_detect_value_columns(df)
    desc_col = try_detect_description_column(
        df, exclude=[c for c in [date_col, value_info.get("single")] if c]
    )

    if date_col is None or desc_col is None or (
        value_info.get("single") is None
        and value_info.get("debit") is None
        and value_info.get("credit") is None
    ):
        return None

    out = pd.DataFrame()
    out["data"] = df[date_col].apply(normalize_date)
    out["descricao"] = df[desc_col].astype(str).str.strip()

    if value_info.get("single"):
        vcol = value_info["single"]
        out["valor"] = df[vcol].apply(parse_brazilian_number)
    else:
        debit_col = value_info.get("debit")
        credit_col = value_info.get("credit")
        debit_series = (
            df[debit_col].apply(parse_brazilian_number) if debit_col else 0.0
        )
        credit_series = (
            df[credit_col].apply(parse_brazilian_number) if credit_col else 0.0
        )
        out["valor"] = credit_series.fillna(0.0) - debit_series.fillna(0.0)

    out = out.dropna(subset=["data", "descricao", "valor"])
    if out.empty:
        return None

    return out.reset_index(drop=True)


def extract_text_from_pdf_with_pdfplumber(file_bytes: bytes) -> str:
    """Extrai texto bruto de um PDF usando pdfplumber."""
    texts: List[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                texts.append(t)
    return "\n".join(texts)


def extract_transactions_with_openai_from_text(text: str) -> pd.DataFrame:
    """
    Usa o modelo da OpenAI para extrair lançamentos (data, descrição, valor)
    a partir do texto bruto de um extrato bancário.
    """
    if not text.strip():
        raise ValueError("Texto do PDF vazio, não é possível extrair lançamentos.")

    client = get_openai_client()

    # Limita o tamanho do texto para evitar estouro de contexto em PDFs muito grandes
    max_chars = 50000
    trimmed_text = text[:max_chars]

    user_prompt = (
        "A seguir está o conteúdo (texto ou CSV) de um extrato bancário brasileiro. "
        "Identifique todos os lançamentos bancários presentes, extraindo para cada um: "
        "data, descrição e valor (positivo para créditos, negativo para débitos). "
        "IMPORTANTE: retorne SOMENTE um JSON no formato:\n"
        '{"lancamentos": [{"data": "...", "descricao": "...", "valor": 0.0}, ...]}\n\n'
        "Conteúdo do extrato:\n"
        f"{trimmed_text}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    content = response.choices[0].message.content
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        raise ValueError(
            "A resposta da OpenAI para extração não pôde ser interpretada como JSON."
        )

    lancamentos_resp = data.get("lancamentos", data)
    if not isinstance(lancamentos_resp, list):
        raise ValueError(
            "O JSON retornado pela OpenAI para extração não contém um array de lançamentos válido."
        )

    records: List[Dict] = []
    for item in lancamentos_resp:
        raw_date = item.get("data")
        raw_desc = item.get("descricao")
        raw_valor = item.get("valor")

        date_str = normalize_date(raw_date) or str(raw_date)
        valor = parse_brazilian_number(raw_valor)

        records.append(
            {
                "data": date_str,
                "descricao": str(raw_desc).strip() if raw_desc is not None else "",
                "valor": valor,
            }
        )

    df = pd.DataFrame(records)
    df = df.dropna(subset=["data", "descricao", "valor"])
    if df.empty:
        raise ValueError(
            "A OpenAI não conseguiu extrair lançamentos válidos do texto do extrato."
        )

    return df.reset_index(drop=True)


def extract_transactions_with_openai_from_images(
    images: List[Image.Image],
) -> pd.DataFrame:
    """
    Usa o modelo de visão da OpenAI (gpt-4o) para extrair lançamentos
    (data, descrição, valor) a partir de imagens de extratos bancários.
    """
    if not images:
        raise ValueError("Lista de imagens vazia, não é possível extrair lançamentos.")

    client = get_openai_client()

    image_contents = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        image_contents.append(
            {
                "type": "input_image",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )

    user_content = [
        {
            "type": "text",
            "text": (
                "Extraia todos os lançamentos financeiros dessas imagens de extrato "
                "bancário brasileiro. Para cada lançamento retorne data, descrição e "
                "valor em formato JSON no formato:\n"
                '{"lancamentos": [{"data": "...", "descricao": "...", "valor": 0.0}, ...]}\n\n'
                "Ignore cabeçalhos, rodapés e informações da conta."
            ),
        },
        *image_contents,
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": user_content}],
        response_format={"type": "json_object"},
        temperature=0,
    )

    content = response.choices[0].message.content
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        raise ValueError(
            "A resposta da OpenAI Vision para extração não pôde ser interpretada como JSON."
        )

    lancamentos_resp = data.get("lancamentos", data)
    if not isinstance(lancamentos_resp, list):
        raise ValueError(
            "O JSON retornado pela OpenAI Vision não contém um array de lançamentos válido."
        )

    records: List[Dict] = []
    for item in lancamentos_resp:
        raw_date = item.get("data")
        raw_desc = item.get("descricao")
        raw_valor = item.get("valor")

        date_str = normalize_date(raw_date) or str(raw_date)
        valor = parse_brazilian_number(raw_valor)

        records.append(
            {
                "data": date_str,
                "descricao": str(raw_desc).strip() if raw_desc is not None else "",
                "valor": valor,
            }
        )

    df = pd.DataFrame(records)
    df = df.dropna(subset=["data", "descricao", "valor"])
    if df.empty:
        raise ValueError(
            "A OpenAI Vision não conseguiu extrair lançamentos válidos das imagens do extrato."
        )

    return df.reset_index(drop=True)


def load_statement(uploaded_file) -> pd.DataFrame:
    """
    Detecta tipo do arquivo e delega para o parser apropriado.

    Para PDF, aplica uma lógica em cascata:
    1) Tenta extrair com pdfplumber
    2) Se falhar, tenta com PyMuPDF (fitz)
    3) Se ainda falhar, extrai o texto bruto e usa a OpenAI para identificar lançamentos
    """
    if uploaded_file is None:
        raise ValueError("Nenhum arquivo enviado.")

    name = uploaded_file.name.lower()

    # -----------------
    # Arquivos CSV
    # -----------------
    if name.endswith(".csv"):
        file_bytes = uploaded_file.read()

        # 1) Tenta ler com pandas usando diferentes separadores
        def parse_csv_bytes(b: bytes) -> Optional[pd.DataFrame]:
            for encoding in ("utf-8", "latin-1"):
                try:
                    text = b.decode(encoding)
                except UnicodeDecodeError:
                    continue
                for sep in [",", ";", "\t"]:
                    try:
                        df_raw = pd.read_csv(io.StringIO(text), sep=sep)
                    except Exception:
                        continue
                    normalized = normalize_statement_dataframe(df_raw)
                    if normalized is not None and not normalized.empty:
                        return normalized
            return None

        df_csv = parse_csv_bytes(file_bytes)
        if df_csv is not None and not df_csv.empty:
            return df_csv

        # 2) Fallback: envia o conteúdo bruto do CSV para a OpenAI extrair os lançamentos
        for encoding in ("utf-8", "latin-1"):
            try:
                csv_text = file_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                csv_text = ""
        if not csv_text:
            raise ValueError(
                "Não foi possível decodificar o CSV nem extrair as colunas automaticamente."
            )

        df_ai_csv = extract_transactions_with_openai_from_text(csv_text)
        return df_ai_csv

    # -----------------
    # Arquivos PDF
    # -----------------
    if name.endswith(".pdf"):
        file_bytes = uploaded_file.read()

        # 1) Tenta extrair texto com pdfplumber
        try:
            raw_text = extract_text_from_pdf_with_pdfplumber(file_bytes)
        except Exception as e:
            raw_text = ""

        if raw_text and len(raw_text) >= 100:
            # Texto suficiente: usa OpenAI para extrair lançamentos do texto
            df_ai_pdf_text = extract_transactions_with_openai_from_text(raw_text)
            return df_ai_pdf_text

        # 2) Possível PDF baseado em imagem: usa pdf2image + OpenAI Vision
        try:
            images: List[Image.Image] = convert_from_bytes(file_bytes)
        except Exception as e:
            raise ValueError(
                "Não foi possível converter o PDF em imagens. "
                "Verifique se o arquivo está corrompido ou tente outro extrato."
            ) from e

        if not images:
            raise ValueError(
                "Nenhuma página de imagem foi gerada a partir do PDF. "
                "Não é possível extrair os lançamentos."
            )

        df_ai_pdf_images = extract_transactions_with_openai_from_images(images)
        return df_ai_pdf_images

    raise ValueError("Tipo de arquivo não suportado. Use PDF ou CSV.")


# -----------------------------------
# Funções de integração com a OpenAI
# -----------------------------------
def get_openai_client() -> OpenAI:
    """
    Cria cliente da OpenAI usando a chave armazenada em st.secrets["OPENAI_API_KEY"].

    A chave não deve aparecer na interface, apenas ser configurada via secrets.
    """
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception as e:
        raise ValueError(
            "OPENAI_API_KEY não encontrada em st.secrets. "
            "Defina-a em .streamlit/secrets.toml ou nas configurações de implantação."
        ) from e

    return OpenAI(api_key=api_key)


def classify_transactions_with_openai(df: pd.DataFrame) -> pd.DataFrame:
    """
    Envia os lançamentos para o modelo gpt-4o-mini para classificação.

    O modelo recebe como entrada um JSON com o array de lançamentos e deve
    retornar apenas um JSON contendo, para cada lançamento:
    data, descricao, valor, categoria, confianca (alta/media/baixa).
    """
    if df.empty:
        raise ValueError("Não há lançamentos para classificar.")

    client = get_openai_client()

    lancamentos = []
    for _, row in df.iterrows():
        lancamentos.append(
            {
                "data": row["data"],
                "descricao": row["descricao"],
                "valor": float(row["valor"]) if row["valor"] is not None else 0.0,
            }
        )

    payload = {"lancamentos": lancamentos}

    user_prompt = (
        "Classifique os seguintes lançamentos bancários conforme o prompt de sistema.\n"
        "IMPORTANTE: retorne SOMENTE um JSON no formato:\n"
        '{"lancamentos": [{"data": "...", "descricao": "...", "valor": 0.0, '
        '"categoria": "...", "confianca": "alta|media|baixa"}, ...]}\n\n'
        "Aqui estão os lançamentos em JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    content = response.choices[0].message.content
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        raise ValueError("A resposta da OpenAI não pôde ser interpretada como JSON.")

    # Aceita tanto {"lancamentos": [...]} quanto um array direto por segurança
    lancamentos_resp = data.get("lancamentos", data)
    if not isinstance(lancamentos_resp, list):
        raise ValueError("O JSON retornado não contém um array de lançamentos válido.")

    result_rows: List[Dict] = []
    for item in lancamentos_resp:
        categoria = str(item.get("categoria", "")).strip()
        confianca_raw = str(item.get("confianca", "")).strip().lower()

        # Normaliza confiança para Alta / Média / Baixa
        if "alta" in confianca_raw:
            confianca = "Alta"
        elif "med" in confianca_raw or "méd" in confianca_raw:
            confianca = "Média"
        elif "baix" in confianca_raw:
            confianca = "Baixa"
        else:
            confianca = "Baixa"

        # Se confiança for baixa e categoria não for Revisar, força categoria Revisar
        if confianca == "Baixa" and categoria.lower() != "revisar":
            categoria = "Revisar"

        result_rows.append(
            {
                "data": item.get("data"),
                "descricao": item.get("descricao"),
                "valor": item.get("valor"),
                "categoria": categoria,
                "confianca": confianca,
            }
        )

    result_df = pd.DataFrame(result_rows)
    return result_df


# -----------------------------
# Funções de estilo e visual
# -----------------------------
def highlight_confidence(row: pd.Series):
    """
    Define a cor de fundo da linha de acordo com a confiança:
    - Verde: Alta
    - Amarelo: Média
    - Vermelho: Baixa
    """
    conf = str(row.get("confianca", "")).lower()
    if "alta" in conf:
        color = "#d4edda"  # verde claro
    elif "méd" in conf or "med" in conf:
        color = "#fff3cd"  # amarelo claro
    elif "baix" in conf:
        color = "#f8d7da"  # vermelho claro
    else:
        color = ""
    return [f"background-color: {color}" if color else "" for _ in row]


def export_to_excel(df: pd.DataFrame) -> bytes:
    """Gera um arquivo Excel em memória a partir do DataFrame."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Lançamentos")
    return output.getvalue()


def plot_category_summary(df: pd.DataFrame):
    """Plota gráfico de barras com total por categoria usando Plotly."""
    if df.empty or "categoria" not in df.columns or "valor" not in df.columns:
        return

    summary = df.groupby("categoria", dropna=False)["valor"].sum().reset_index()
    summary = summary.sort_values("valor", ascending=False)

    fig = px.bar(
        summary,
        x="categoria",
        y="valor",
        title="Total por Categoria",
        labels={"categoria": "Categoria", "valor": "Total (R$)"},
        text_auto=".2f",
    )
    fig.update_layout(xaxis_title="Categoria", yaxis_title="Total (R$)")
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Interface principal Streamlit
# -----------------------------
def main():
    # Barra lateral apenas com upload de arquivo, sem menção a configurações ou chave
    st.sidebar.title("ConciBot")
    uploaded_file = st.sidebar.file_uploader(
        "Faça upload do seu extrato bancário em PDF ou CSV", type=["pdf", "csv"]
    )

    st.title("ConciBot 🧾")
    st.markdown("Faça upload do seu extrato bancário em PDF ou CSV.")

    # Estado da sessão para armazenar dados entre interações
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = None
    if "classified_df" not in st.session_state:
        st.session_state.classified_df = None
    if "edited_df" not in st.session_state:
        st.session_state.edited_df = None

    # ---------------------------
    # Etapa 1: leitura do arquivo
    # ---------------------------
    if uploaded_file is not None:
        try:
            df = load_statement(uploaded_file)
            st.session_state.raw_df = df
            st.session_state.classified_df = None
            st.session_state.edited_df = None

            st.subheader("Lançamentos extraídos")
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")
    else:
        st.info("Envie um extrato bancário em PDF ou CSV na barra lateral para começar.")

    # --------------------------------------
    # Etapa 2: classificação automática com a OpenAI
    # --------------------------------------
    if st.session_state.raw_df is not None:
        st.markdown("### Classificação com IA")

        # Classifica automaticamente assim que os dados forem carregados
        if st.session_state.classified_df is None:
            with st.spinner("Classificando lançamentos com IA..."):
                try:
                    classified_df = classify_transactions_with_openai(
                        st.session_state.raw_df
                    )
                    st.session_state.classified_df = classified_df
                    st.session_state.edited_df = classified_df.copy()
                    st.success("Classificação concluída com sucesso!")
                except Exception as e:
                    st.error(f"Erro ao classificar lançamentos: {e}")

    # ---------------------------------
    # Etapa 3: visualização e edição
    # ---------------------------------
    if st.session_state.classified_df is not None:
        st.markdown("### Resultados da classificação")

        current_df = (
            st.session_state.edited_df
            if st.session_state.edited_df is not None
            else st.session_state.classified_df
        ).copy()

        # Tabela colorida por confiança (visual)
        styled = current_df.style.apply(highlight_confidence, axis=1)
        st.markdown("**Tabela colorida por nível de confiança:**")
        try:
            # Exibe tabela estática com cores
            st.table(styled)
        except Exception:
            # Se por algum motivo o Styler não funcionar, cai para a tabela simples
            st.dataframe(current_df, use_container_width=True)

        # Editor interativo para permitir edição de categorias
        st.markdown("**Edite as categorias diretamente na tabela abaixo, se necessário:**")
        edited_df = st.data_editor(
            current_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "data": st.column_config.TextColumn("Data", disabled=True),
                "descricao": st.column_config.TextColumn("Descrição", disabled=True),
                "valor": st.column_config.NumberColumn("Valor (R$)", disabled=True),
                "categoria": st.column_config.TextColumn("Categoria"),
                "confianca": st.column_config.TextColumn("Confiança", disabled=True),
            },
            key="editor",
        )
        st.session_state.edited_df = edited_df

        # ---------------------------------
        # Etapa 4: exportação e gráficos
        # ---------------------------------
        st.markdown("### Exportação e resumo")
        col1, col2 = st.columns(2)

        with col1:
            if st.session_state.edited_df is not None:
                excel_bytes = export_to_excel(st.session_state.edited_df)
                st.download_button(
                    "Exportar Excel",
                    data=excel_bytes,
                    file_name="concibot_lancamentos_classificados.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                )

        with col2:
            if st.session_state.edited_df is not None:
                st.markdown("**Resumo visual (total por categoria):**")
                plot_category_summary(st.session_state.edited_df)


if __name__ == "__main__":
    main()

