import io
import json
import re
from typing import List, Dict, Optional

import fitz  # PyMuPDF
import pandas as pd
import pdfplumber
import plotly.express as px
import streamlit as st
from openai import OpenAI


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


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Encontra coluna aproximada com base em palavras-chave (case-insensitive)."""
    lower_map = {col: col.lower() for col in df.columns}
    for target in candidates:
        for col, lower in lower_map.items():
            if target in lower:
                return col
    return None


def parse_csv_file(uploaded_file) -> pd.DataFrame:
    """Lê arquivos CSV de extrato bancário em diferentes formatações comuns."""
    # Tentativa com separador ';' (comum em bancos brasileiros)
    try:
        df = pd.read_csv(uploaded_file, sep=";", encoding="latin-1")
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)

    # Procura colunas típicas de extrato (data, descrição, valor)
    data_col = find_column(df, ["data"])
    desc_col = find_column(df, ["descri", "hist", "lança", "histor"])
    valor_col = find_column(df, ["valor", "val"])

    if not all([data_col, desc_col, valor_col]):
        raise ValueError(
            "Não foi possível identificar automaticamente as colunas de data, "
            "descrição e valor no CSV."
        )

    out = pd.DataFrame()
    out["data"] = df[data_col].apply(normalize_date)
    out["descricao"] = df[desc_col].astype(str).str.strip()
    out["valor"] = df[valor_col].apply(parse_brazilian_number)

    out = out.dropna(subset=["data", "descricao", "valor"])
    return out.reset_index(drop=True)


def extract_transactions_from_lines(text_lines: List[str]) -> pd.DataFrame:
    """
    Tenta extrair lançamentos a partir de linhas de texto usando expressões regulares.

    Projetado para lidar com formatos comuns de extratos brasileiros, como:
    01/02/2025 COMPRA SUPERMERCADO -123,45
    01-02-2025 TED RECEBIDA 1.234,56
    """
    # Data (dd/mm/aaaa ou dd-mm-aaaa), descrição no meio e valor no final
    pattern = re.compile(
        r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\s+(.+?)\s+(-?\d+[\.\d]*,\d{2}|-?\d+,\d{2}|-?\d+\.\d{2})$"
    )

    records: List[Dict] = []
    for line in text_lines:
        match = pattern.search(line)
        if not match:
            continue
        raw_date, raw_desc, raw_value = match.groups()
        date_str = normalize_date(raw_date) or raw_date
        value = parse_brazilian_number(raw_value)
        records.append(
            {
                "data": date_str,
                "descricao": raw_desc.strip(),
                "valor": value,
            }
        )

    df = pd.DataFrame(records)
    df = df.dropna(subset=["data", "descricao", "valor"])
    return df.reset_index(drop=True)


def extract_text_lines_with_pdfplumber(file_bytes: bytes) -> List[str]:
    """Extrai linhas de texto de um PDF usando pdfplumber."""
    text_lines: List[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if line:
                    text_lines.append(line)
    return text_lines


def extract_text_lines_with_pymupdf(file_bytes: bytes) -> List[str]:
    """Extrai linhas de texto de um PDF usando PyMuPDF (fitz)."""
    text_lines: List[str] = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text = page.get_text("text")
            if not text:
                continue
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if line:
                    text_lines.append(line)
    return text_lines


def extract_full_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extrai texto bruto do PDF, tentando primeiro PyMuPDF e depois pdfplumber.
    Útil para enviar o texto completo para a OpenAI quando o layout é complexo.
    """
    # Primeiro tenta PyMuPDF (geralmente mais robusto para Nubank e outros)
    try:
        texts: List[str] = []
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                t = page.get_text("text")
                if t:
                    texts.append(t)
        if texts:
            return "\n".join(texts)
    except Exception:
        pass

    # Fallback para pdfplumber
    try:
        texts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    texts.append(t)
        if texts:
            return "\n".join(texts)
    except Exception:
        pass

    return ""


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
        "A seguir está o texto completo (ou parte) de um extrato bancário brasileiro. "
        "Identifique todos os lançamentos bancários presentes, extraindo para cada um: "
        "data, descrição e valor (positivo para créditos, negativo para débitos). "
        "IMPORTANTE: retorne SOMENTE um JSON no formato:\n"
        '{"lancamentos": [{"data": "...", "descricao": "...", "valor": 0.0}, ...]}\n\n'
        "Texto do extrato:\n"
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

    if name.endswith(".csv"):
        return parse_csv_file(uploaded_file)

    if name.endswith(".pdf"):
        # Lê o conteúdo do arquivo em memória para reutilizar nas diferentes estratégias
        file_bytes = uploaded_file.read()

        # 1) pdfplumber
        try:
            lines = extract_text_lines_with_pdfplumber(file_bytes)
            df_pdfplumber = extract_transactions_from_lines(lines)
            if not df_pdfplumber.empty:
                return df_pdfplumber
        except Exception:
            pass

        # 2) PyMuPDF (fitz)
        try:
            lines = extract_text_lines_with_pymupdf(file_bytes)
            df_pymupdf = extract_transactions_from_lines(lines)
            if not df_pymupdf.empty:
                return df_pymupdf
        except Exception:
            pass

        # 3) OpenAI: extrai texto bruto e pede para a IA identificar lançamentos
        try:
            raw_text = extract_full_text_from_pdf(file_bytes)
            if not raw_text.strip():
                raise ValueError("Texto do PDF vazio, não é possível usar OpenAI.")

            df_ai = extract_transactions_with_openai_from_text(raw_text)
            if not df_ai.empty:
                return df_ai
        except Exception as e:
            raise ValueError(
                "Não foi possível extrair lançamentos do PDF, mesmo após tentar pdfplumber, "
                "PyMuPDF e OpenAI. Verifique se o extrato está legível e tente novamente."
            ) from e

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
    # Etapa 2: classificação com a OpenAI
    # --------------------------------------
    if st.session_state.raw_df is not None:
        st.markdown("### Classificação com IA")

        if st.button("Classificar lançamentos com IA", type="primary"):
            with st.spinner("Enviando lançamentos para a OpenAI..."):
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

