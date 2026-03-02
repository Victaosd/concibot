import base64
import io
import json
import re
from typing import List, Dict, Optional

import fitz  # PyMuPDF
import pandas as pd
import pdfplumber
import plotly.express as px
import plotly.graph_objects as go
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

# -----------------------------
# Paleta de cores por categoria
# -----------------------------
CATEGORY_COLORS = {
    "Receita": "#00C48C",
    "Transferência Pessoal": "#6C8EBF",
    "Transferência Própria": "#A0A0A0",
    "Fatura Cartão": "#FF6B6B",
    "Investimento": "#FFD93D",
    "Internet e Telecom": "#4ECDC4",
    "Apostas e Jogos": "#FF4757",
    "Alimentação": "#FFA94D",
    "Transporte": "#74B9FF",
    "Moradia": "#A8E6CF",
    "Saúde": "#FF8B94",
    "Serviço Financeiro": "#9B59B6",
    "Impostos e Taxas": "#E17055",
    "Folha de Pagamento": "#00B894",
    "Fornecedores": "#FDCB6E",
    "Outros": "#B2BEC3",
}

# ---------------------------------------
# Prompts do sistema para a IA
# ---------------------------------------
SYSTEM_PROMPT = (
    "Você é um contador brasileiro sênior com 20 anos de experiência em escritórios de contabilidade. "
    "Analise cada lançamento bancário e classifique com máxima precisão em UMA das categorias abaixo.\n\n"
    "REGRAS OBRIGATÓRIAS:\n"
    "- NUNCA classifique como Transferência Pessoal apenas porque tem a palavra Pix ou TED — analise o destinatário.\n"
    "- Se o remetente/destinatário for o MESMO TITULAR da conta (mesmo CPF ou mesmo nome): use Transferência Própria.\n"
    "- Se for Resgate RDB, CDB, Aplicação RDB, Tesouro, fundos: use Investimento.\n"
    "- Se for Betboom, Bet365, Betano, Sportingbet, PAY4FUN, OKTO IP ou casas de aposta: use Apostas e Jogos.\n"
    "- Se for Alares, Claro, Vivo, Tim, Oi ou provedores de internet: use Internet e Telecom.\n"
    "- Se for 'Pagamento de fatura': use Fatura Cartão.\n"
    "- Se o destinatário for pessoa física diferente do titular: use Transferência Pessoal.\n"
    "- Se for entrada de terceiros (clientes, empresas pagando): use Receita.\n\n"
    "CATEGORIAS DISPONÍVEIS:\n"
    "- Receita\n"
    "- Transferência Pessoal\n"
    "- Transferência Própria\n"
    "- Fatura Cartão\n"
    "- Investimento\n"
    "- Internet e Telecom\n"
    "- Apostas e Jogos\n"
    "- Alimentação\n"
    "- Transporte\n"
    "- Moradia\n"
    "- Saúde\n"
    "- Serviço Financeiro\n"
    "- Impostos e Taxas\n"
    "- Folha de Pagamento\n"
    "- Fornecedores\n"
    "- Outros\n\n"
    "Retorne APENAS JSON válido, sem texto nem markdown:\n"
    '[{"data": "...", "descricao": "...", "valor": 0.0, "categoria": "...", '
    '"confianca": "alta|media|baixa", "motivo": "explicação curta"}]'
)

EXTRACTION_SYSTEM_PROMPT = (
    "Você é um assistente contábil brasileiro especializado em extrair "
    "lançamentos bancários de textos de extratos de bancos brasileiros "
    "(Nubank, Itaú, Bradesco, Santander, Banco do Brasil, Caixa, etc.). "
    "Sua tarefa é identificar todos os lançamentos contendo: data, descrição "
    "e valor (positivo para créditos, negativo para débitos). "
    "NÃO inclua linhas de totais, cabeçalhos ou rodapés — apenas lançamentos individuais. "
    "Retorne apenas um JSON com o campo 'lancamentos', que é um array de objetos "
    'no formato: {"data": "DD/MM/YYYY", "descricao": "...", "valor": 0.0}. '
    "O valor deve ser um número decimal positivo para crédito e negativo para débito."
)


# ----------------------------------------
# Funções auxiliares de parsing e limpeza
# ----------------------------------------
def parse_brazilian_number(value) -> Optional[float]:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    s = s.replace("R$", "").replace(" ", "")
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def format_brl(amount) -> str:
    if amount is None or (isinstance(amount, float) and pd.isna(amount)):
        return "R$ 0,00"
    s = f"{float(amount):,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"


def normalize_date(value) -> Optional[str]:
    try:
        dt = pd.to_datetime(value, dayfirst=True, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.date().isoformat()
    except Exception:
        return None


def try_detect_date_column(df: pd.DataFrame) -> Optional[str]:
    date_keywords = ["data", "date", "dt", "transaction date", "posting date", "fecha"]
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
            ratio += 0.2
        if ratio > best_score and ratio >= 0.5:
            best_score = ratio
            best_col = col
    return best_col


def try_detect_value_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
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
    numeric_candidates.sort(key=lambda x: x["score"], reverse=True)
    if len(numeric_candidates) == 1:
        return {"single": numeric_candidates[0]["col"], "debit": None, "credit": None}
    debit_col = None
    credit_col = None
    for cand in numeric_candidates[:3]:
        if debit_col is None and any(k in cand["name"] for k in debit_keywords):
            debit_col = cand["col"]
        if credit_col is None and any(k in cand["name"] for k in credit_keywords):
            credit_col = cand["col"]
    if debit_col or credit_col:
        return {"single": None, "debit": debit_col, "credit": credit_col}
    return {"single": numeric_candidates[0]["col"], "debit": None, "credit": None}


def try_detect_description_column(df: pd.DataFrame, exclude: List[str]) -> Optional[str]:
    desc_keywords = ["descri", "hist", "lança", "histor", "memo", "description", "detalhe"]
    best_col = None
    best_score = 0.0
    for col in df.columns:
        if col in exclude:
            continue
        series = df[col]
        sample = series.dropna().astype(str).head(100)
        if sample.empty:
            continue
        numeric_like = sample.apply(lambda v: parse_brazilian_number(v) is not None).sum()
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
        out["valor"] = df[value_info["single"]].apply(parse_brazilian_number)
    else:
        debit_col = value_info.get("debit")
        credit_col = value_info.get("credit")
        debit_series = df[debit_col].apply(parse_brazilian_number) if debit_col else 0.0
        credit_series = df[credit_col].apply(parse_brazilian_number) if credit_col else 0.0
        out["valor"] = credit_series.fillna(0.0) - debit_series.fillna(0.0)
    out = out.dropna(subset=["data", "descricao", "valor"])
    # Remove duplicatas exatas
    out = out.drop_duplicates(subset=["data", "descricao", "valor"])
    if out.empty:
        return None
    return out.reset_index(drop=True)


def extract_text_from_pdf_with_pdfplumber(file_bytes: bytes) -> str:
    texts: List[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                texts.append(t)
    return "\n".join(texts)


def extract_text_from_pdf_with_pymupdf(file_bytes: bytes) -> str:
    texts: List[str] = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            t = page.get_text("text")
            if t:
                texts.append(t)
    return "\n".join(texts)


def extract_transactions_with_openai_from_text(text: str) -> pd.DataFrame:
    if not text.strip():
        raise ValueError("Texto do PDF vazio.")
    client = get_openai_client()
    max_chars = 50000
    trimmed_text = text[:max_chars]
    user_prompt = (
        "A seguir está o conteúdo de um extrato bancário brasileiro. "
        "Identifique TODOS os lançamentos individuais (não inclua totais, cabeçalhos ou rodapés). "
        "Para cada lançamento extraia: data, descrição e valor (positivo=crédito, negativo=débito). "
        "Retorne SOMENTE JSON no formato:\n"
        '{"lancamentos": [{"data": "DD/MM/YYYY", "descricao": "...", "valor": 0.0}]}\n\n'
        f"Conteúdo:\n{trimmed_text}"
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
    data = json.loads(content)
    lancamentos_resp = data.get("lancamentos", data)
    if not isinstance(lancamentos_resp, list):
        raise ValueError("JSON da OpenAI não contém lista de lançamentos válida.")
    records = []
    for item in lancamentos_resp:
        date_str = normalize_date(item.get("data")) or str(item.get("data"))
        valor = parse_brazilian_number(item.get("valor"))
        desc = str(item.get("descricao", "")).strip()
        if desc and valor is not None:
            records.append({"data": date_str, "descricao": desc, "valor": valor})
    df = pd.DataFrame(records).dropna(subset=["data", "descricao", "valor"])
    df = df.drop_duplicates(subset=["data", "descricao", "valor"])
    if df.empty:
        raise ValueError("OpenAI não extraiu lançamentos válidos do texto.")
    return df.reset_index(drop=True)


def extract_transactions_with_openai_from_images(images: List[Image.Image]) -> pd.DataFrame:
    if not images:
        raise ValueError("Lista de imagens vazia.")
    client = get_openai_client()
    image_contents = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        image_contents.append({
            "type": "input_image",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    user_content = [
        {
            "type": "text",
            "text": (
                "Este é um extrato bancário brasileiro. Extraia TODOS os lançamentos individuais "
                "(não inclua linhas de total, cabeçalho ou rodapé). "
                "Retorne JSON com array de objetos: "
                '[{"data": "DD/MM/YYYY", "descricao": "...", "valor": 0.0}] '
                "onde valor é positivo para crédito e negativo para débito."
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
    data = json.loads(content)
    lancamentos_resp = data.get("lancamentos", data)
    if not isinstance(lancamentos_resp, list):
        raise ValueError("JSON da OpenAI Vision não contém lista válida.")
    records = []
    for item in lancamentos_resp:
        date_str = normalize_date(item.get("data")) or str(item.get("data"))
        valor = parse_brazilian_number(item.get("valor"))
        desc = str(item.get("descricao", "")).strip()
        if desc and valor is not None:
            records.append({"data": date_str, "descricao": desc, "valor": valor})
    df = pd.DataFrame(records).dropna(subset=["data", "descricao", "valor"])
    df = df.drop_duplicates(subset=["data", "descricao", "valor"])
    if df.empty:
        raise ValueError("OpenAI Vision não extraiu lançamentos válidos.")
    return df.reset_index(drop=True)


def extrair_lancamentos(arquivo) -> pd.DataFrame:
    if arquivo is None:
        raise ValueError("Nenhum arquivo enviado.")
    name = (getattr(arquivo, "name", "") or "").lower()
    file_bytes = arquivo.read()
    if not file_bytes:
        raise ValueError("Arquivo vazio.")

    if name.endswith(".csv"):
        def parse_csv_bytes(b: bytes) -> Optional[pd.DataFrame]:
            for encoding in ("utf-8", "latin-1"):
                try:
                    text = b.decode(encoding)
                except UnicodeDecodeError:
                    continue
                for sep in [",", ";"]:
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
            df_out = df_csv
        else:
            csv_text = ""
            for encoding in ("utf-8", "latin-1"):
                try:
                    csv_text = file_bytes.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            if not csv_text:
                raise ValueError("Não foi possível decodificar o CSV.")
            df_out = extract_transactions_with_openai_from_text(csv_text)

    elif name.endswith(".pdf"):
        try:
            raw_text = extract_text_from_pdf_with_pdfplumber(file_bytes)
        except Exception:
            raw_text = ""
        text_ok = raw_text.strip() if raw_text else ""
        if len(text_ok) < 100:
            try:
                raw_text_pymupdf = extract_text_from_pdf_with_pymupdf(file_bytes)
            except Exception:
                raw_text_pymupdf = ""
            if len((raw_text_pymupdf or "").strip()) >= 100:
                text_ok = raw_text_pymupdf.strip()
        if text_ok and len(text_ok) >= 100:
            df_out = extract_transactions_with_openai_from_text(text_ok)
        else:
            try:
                images = convert_from_bytes(file_bytes)
            except Exception as e:
                raise ValueError("Não foi possível converter o PDF em imagens.") from e
            if not images:
                raise ValueError("Nenhuma página de imagem gerada a partir do PDF.")
            df_out = extract_transactions_with_openai_from_images(images)
    else:
        raise ValueError("Tipo de arquivo não suportado. Use PDF ou CSV.")

    if df_out is None or df_out.empty:
        raise ValueError("Nenhum lançamento identificado no arquivo.")
    if not {"data", "descricao", "valor"}.issubset(df_out.columns):
        raise ValueError("Colunas esperadas não encontradas: data, descricao, valor.")
    return df_out[["data", "descricao", "valor"]].copy().reset_index(drop=True)


# -----------------------------------
# Funções de integração com a OpenAI
# -----------------------------------
def get_openai_client() -> OpenAI:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception as e:
        raise ValueError(
            "OPENAI_API_KEY não encontrada em st.secrets. "
            "Defina-a nas configurações de implantação."
        ) from e
    return OpenAI(api_key=api_key)


def classify_transactions_with_openai(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Não há lançamentos para classificar.")
    client = get_openai_client()
    total = len(df)
    batch_size = 20
    progress_bar = st.progress(0)
    status_text = st.empty()
    result_rows: List[Dict] = []

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_df = df.iloc[start:end]
        status_text.text(f"Classificando lançamentos {start + 1}–{end} de {total}...")
        progress_bar.progress(int((end / total) * 100))

        lancamentos_batch = []
        for _, row in batch_df.iterrows():
            lancamentos_batch.append({
                "data": row["data"],
                "descricao": row["descricao"],
                "valor": float(row["valor"]) if row["valor"] is not None else 0.0,
            })

        user_prompt = (
            "Classifique os lançamentos bancários abaixo.\n"
            "Retorne APENAS JSON válido, sem texto nem markdown:\n"
            '[{"data":"...","descricao":"...","valor":0.0,"categoria":"...","confianca":"alta|media|baixa","motivo":"..."}]\n\n'
            f"Lançamentos:\n{json.dumps(lancamentos_batch, ensure_ascii=False)}"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )

        content = response.choices[0].message.content
        # Remove markdown se vier
        content = re.sub(r"```json|```", "", content).strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError("Resposta da OpenAI não é JSON válido.")

        lancamentos_resp = data.get("lancamentos", data) if isinstance(data, dict) else data
        if not isinstance(lancamentos_resp, list):
            raise ValueError("JSON da OpenAI não contém lista de lançamentos.")

        for item in lancamentos_resp:
            confianca_raw = str(item.get("confianca", "")).strip().lower()
            if "alta" in confianca_raw:
                confianca = "Alta"
            elif "med" in confianca_raw or "méd" in confianca_raw:
                confianca = "Média"
            else:
                confianca = "Baixa"
            result_rows.append({
                "data": item.get("data"),
                "descricao": item.get("descricao"),
                "valor": item.get("valor"),
                "categoria": str(item.get("categoria", "")).strip(),
                "confianca": confianca,
                "motivo": str(item.get("motivo", "")).strip(),
            })

    progress_bar.progress(100)
    status_text.empty()
    return pd.DataFrame(result_rows)


# -----------------------------
# Funções visuais e de resumo
# -----------------------------
def get_row_style(categoria: str) -> str:
    color = CATEGORY_COLORS.get(categoria, "#B2BEC3")
    return color


def compute_financial_summary(df: pd.DataFrame):
    if df is None or df.empty:
        return 0, 0.0, 0.0, 0.0
    total_lanc = len(df)
    receitas = df.loc[df["valor"] > 0, "valor"].sum()
    despesas = df.loc[df["valor"] < 0, "valor"].sum()
    saldo = df["valor"].sum()
    return total_lanc, float(receitas), float(despesas), float(saldo)


def export_to_excel(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Aba principal
        export_df = df.copy()
        if "valor" in export_df.columns:
            export_df["valor_formatado"] = export_df["valor"].apply(format_brl)
        export_df.to_excel(writer, index=False, sheet_name="Lançamentos")

        # Aba resumo por categoria
        if "categoria" in df.columns and "valor" in df.columns:
            resumo = df.groupby("categoria").agg(
                quantidade=("valor", "count"),
                total=("valor", "sum")
            ).reset_index()
            resumo["total_formatado"] = resumo["total"].apply(format_brl)
            resumo.to_excel(writer, index=False, sheet_name="Resumo por Categoria")

        # Aba alertas (baixa confiança)
        if "confianca" in df.columns:
            alertas = df[df["confianca"] == "Baixa"]
            if not alertas.empty:
                alertas.to_excel(writer, index=False, sheet_name="Revisar")

    return output.getvalue()


def plot_charts(df: pd.DataFrame):
    if df.empty or "categoria" not in df.columns:
        return

    col1, col2, col3 = st.columns(3)

    # Gráfico 1: Pizza por categoria
    with col1:
        summary = (
            df.assign(valor_abs=df["valor"].abs())
            .groupby("categoria")["valor_abs"]
            .sum()
            .reset_index()
        )
        colors = [CATEGORY_COLORS.get(c, "#B2BEC3") for c in summary["categoria"]]
        fig_pie = px.pie(
            summary,
            names="categoria",
            values="valor_abs",
            title="Distribuição por categoria",
            color="categoria",
            color_discrete_map=CATEGORY_COLORS,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Gráfico 2: Receitas vs Despesas por semana
    with col2:
        df_time = df.copy()
        df_time["data_dt"] = pd.to_datetime(df_time["data"], errors="coerce")
        df_time["semana"] = df_time["data_dt"].dt.to_period("W").astype(str)
        df_time["tipo"] = df_time["valor"].apply(lambda v: "Receita" if v > 0 else "Despesa")
        df_time["valor_abs"] = df_time["valor"].abs()
        semanal = df_time.groupby(["semana", "tipo"])["valor_abs"].sum().reset_index()
        fig_bar = px.bar(
            semanal,
            x="semana",
            y="valor_abs",
            color="tipo",
            barmode="group",
            title="Receitas vs Despesas por semana",
            color_discrete_map={"Receita": "#00C48C", "Despesa": "#FF6B6B"},
        )
        fig_bar.update_layout(height=350, xaxis_title="", yaxis_title="R$")
        st.plotly_chart(fig_bar, use_container_width=True)

    # Gráfico 3: Evolução do saldo
    with col3:
        df_saldo = df.copy()
        df_saldo["data_dt"] = pd.to_datetime(df_saldo["data"], errors="coerce")
        df_saldo = df_saldo.sort_values("data_dt")
        df_saldo["saldo_acum"] = df_saldo["valor"].cumsum()
        fig_line = px.line(
            df_saldo,
            x="data_dt",
            y="saldo_acum",
            title="Evolução do saldo",
            color_discrete_sequence=["#6C8EBF"],
        )
        fig_line.update_layout(height=350, xaxis_title="", yaxis_title="R$")
        st.plotly_chart(fig_line, use_container_width=True)


# -----------------------------
# Interface principal Streamlit
# -----------------------------
def main():
    # CSS customizado para badges coloridos
    st.markdown("""
    <style>
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
        color: white;
        text-shadow: 0px 0px 2px rgba(0,0,0,0.4);
    }
    .card-metric {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    # Barra lateral
    st.sidebar.title("ConciBot 🧾")
    st.sidebar.caption("Classificação inteligente de extratos bancários")
    uploaded_file = st.sidebar.file_uploader(
        "Faça upload do seu extrato bancário em PDF ou CSV",
        type=["pdf", "csv"]
    )

    # Header principal
    st.markdown("# 📊 ConciBot")
    st.caption("Classificação inteligente de extratos bancários com IA")
    st.divider()

    # Estado da sessão
    for key in ["raw_df", "classified_df", "edited_df", "last_file_name"]:
        if key not in st.session_state:
            st.session_state[key] = None

    # Etapa 1: leitura do arquivo
    if uploaded_file is not None:
        file_name = uploaded_file.name

        # Só reprocessa se for um arquivo diferente
        if st.session_state.last_file_name != file_name:
            try:
                with st.spinner("📂 Lendo e extraindo lançamentos..."):
                    df = extrair_lancamentos(uploaded_file)
                st.session_state.raw_df = df
                st.session_state.classified_df = None
                st.session_state.edited_df = None
                st.session_state.last_file_name = file_name
            except Exception as e:
                st.error(f"❌ Erro ao processar o arquivo: {e}")
                return

        if st.session_state.raw_df is not None:
            with st.expander(f"📋 Lançamentos extraídos ({len(st.session_state.raw_df)} itens)", expanded=False):
                st.dataframe(st.session_state.raw_df, use_container_width=True)

        # Etapa 2: classificação automática
        if st.session_state.classified_df is None and st.session_state.raw_df is not None:
            try:
                with st.spinner("🤖 Classificando lançamentos com IA..."):
                    classified_df = classify_transactions_with_openai(st.session_state.raw_df)
                st.session_state.classified_df = classified_df
                st.session_state.edited_df = classified_df.copy()
                st.success("✅ Classificação concluída com sucesso!")
            except Exception as e:
                st.error(f"❌ Erro ao classificar lançamentos: {e}")
                return

    else:
        st.info("📤 Envie um extrato bancário em PDF ou CSV na barra lateral para começar.")
        return

    # Etapa 3: visualização
    if st.session_state.classified_df is not None:
        current_df = (
            st.session_state.edited_df
            if st.session_state.edited_df is not None
            else st.session_state.classified_df
        ).copy()

        # Alerta de apostas
        if "categoria" in current_df.columns:
            apostas_df = current_df[current_df["categoria"] == "Apostas e Jogos"]
            if not apostas_df.empty:
                total_apostas = apostas_df["valor"].abs().sum()
                st.warning(
                    f"⚠️ **Atenção:** {len(apostas_df)} lançamento(s) de apostas detectado(s) "
                    f"totalizando **{format_brl(total_apostas)}**"
                )

        # Cards de resumo
        total_lanc, total_receitas, total_despesas, saldo = compute_financial_summary(current_df)
        st.markdown("### 📊 Resumo do período")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📌 Total de lançamentos", f"{total_lanc}")
        c2.metric("📈 Total de receitas", format_brl(total_receitas))
        c3.metric("📉 Total de despesas", format_brl(abs(total_despesas)))
        saldo_delta = "positivo" if saldo >= 0 else "negativo"
        c4.metric("💰 Saldo do período", format_brl(saldo))

        st.divider()

        # Filtros
        st.markdown("### 🔍 Filtros")
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            categorias_disponiveis = ["Todas"] + sorted(current_df["categoria"].dropna().unique().tolist())
            filtro_categoria = st.selectbox("Filtrar por categoria:", categorias_disponiveis)
        with col_f2:
            filtro_busca = st.text_input("Buscar por descrição:", placeholder="Digite parte da descrição...")

        # Aplica filtros
        filtered_df = current_df.copy()
        if filtro_categoria != "Todas":
            filtered_df = filtered_df[filtered_df["categoria"] == filtro_categoria]
        if filtro_busca:
            filtered_df = filtered_df[
                filtered_df["descricao"].str.contains(filtro_busca, case=False, na=False)
            ]

        st.caption(f"Exibindo {len(filtered_df)} de {len(current_df)} lançamentos")

        # Tabela com badge de categoria e ícone de confiança
        st.markdown("### 📋 Lançamentos classificados")

        display_rows = []
        for _, row in filtered_df.iterrows():
            cat = row.get("categoria", "Outros")
            color = CATEGORY_COLORS.get(cat, "#B2BEC3")
            conf = row.get("confianca", "Baixa")
            conf_icon = "✅" if conf == "Alta" else ("⚠️" if conf == "Média" else "❓")
            valor = row.get("valor", 0)
            valor_fmt = format_brl(valor)
            display_rows.append({
                "Data": row.get("data", ""),
                "Descrição": row.get("descricao", ""),
                "Valor": valor_fmt,
                "Categoria": cat,
                "Confiança": f"{conf_icon} {conf}",
            })

        display_df_show = pd.DataFrame(display_rows)

        def color_categoria(val):
            color = CATEGORY_COLORS.get(val, "#B2BEC3")
            return f"background-color: {color}22; color: {color}; font-weight: bold; border-radius: 6px; padding: 2px 6px;"

        def color_valor(val):
            if val.startswith("R$ -"):
                return "color: #FF6B6B; font-weight: bold;"
            return "color: #00C48C; font-weight: bold;"

        styled = (
            display_df_show.style
            .applymap(color_categoria, subset=["Categoria"])
            .applymap(color_valor, subset=["Valor"])
        )

        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Editor para corrigir categorias
        st.markdown("### ✏️ Editar categorias")
        st.caption("Clique em uma célula da coluna Categoria para corrigir se necessário.")

        edited_df = st.data_editor(
            current_df[["data", "descricao", "valor", "categoria"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "data": st.column_config.TextColumn("Data", disabled=True),
                "descricao": st.column_config.TextColumn("Descrição", disabled=True),
                "valor": st.column_config.NumberColumn("Valor", disabled=True, format="R$ %.2f"),
                "categoria": st.column_config.SelectboxColumn(
                    "Categoria",
                    options=list(CATEGORY_COLORS.keys()),
                ),
            },
            key="editor",
        )
        # Preserva colunas extras ao salvar edição
        if edited_df is not None:
            merged = current_df.copy()
            merged["categoria"] = edited_df["categoria"].values
            st.session_state.edited_df = merged

        st.divider()

        # Gráficos
        st.markdown("### 📈 Gráficos")
        final_df = st.session_state.edited_df if st.session_state.edited_df is not None else current_df
        plot_charts(final_df)

        st.divider()

        # Exportação
        st.markdown("### 💾 Exportar")
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            excel_bytes = export_to_excel(final_df)
            st.download_button(
                "📊 Exportar Excel",
                data=excel_bytes,
                file_name="concibot_classificado.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True,
            )
        with col_e2:
            csv_export = final_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📄 Exportar CSV",
                data=csv_export,
                file_name="concibot_classificado.csv",
                mime="text/csv",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()