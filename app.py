import base64
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
    "Você é um contador brasileiro sênior com 20 anos de experiência. "
    "Analise cada lançamento e classifique com máxima precisão em UMA das seguintes "
    "categorias, usando o nome da empresa, CNPJ e contexto para classificar:\n\n"
    "- Receita: entradas de clientes, vendas, serviços prestados\n"
    "- Transferência Pessoal: PIX/TED entre pessoas físicas conhecidas\n"
    "- Fatura Cartão: pagamento de fatura de cartão de crédito\n"
    "- Investimento: RDB, CDB, Tesouro Direto, ações, fundos, resgates\n"
    "- Internet e Telecom: provedores internet, telefonia, streaming\n"
    "- Apostas e Jogos: casas de aposta, jogos online (Betboom, Bet365, Betano, Sportingbet, etc.)\n"
    "- Alimentação: restaurantes, delivery, supermercados, padarias\n"
    "- Transporte: Uber, 99, combustível, pedágio, estacionamento\n"
    "- Moradia: aluguel, condomínio, água, luz, gás\n"
    "- Saúde: farmácias, planos de saúde, consultas médicas\n"
    "- Serviço Financeiro: fintechs, corretoras, instituições de pagamento\n"
    "- Impostos e Taxas: INSS, FGTS, IPTU, IPVA, DAS, IOF, tributos\n"
    "- Folha de Pagamento: salários, pró-labore, benefícios\n"
    "- Fornecedores: pagamentos a empresas por produtos ou serviços\n"
    "- Outros: apenas se absolutamente impossível classificar\n\n"
    "REGRAS:\n"
    "- NUNCA classifique como Transferência apenas porque aparece PIX ou TED; analise o destinatário.\n"
    "- Se for transferência entre pessoas físicas conhecidas: use Transferência Pessoal.\n"
    "- Se o lançamento envolver Betboom, Bet365, Betano, Sportingbet ou casas de aposta: use Apostas e Jogos.\n"
    "- Se envolver RDB, CDB, aplicação, resgate, Tesouro Direto, fundos ou ações: use Investimento.\n"
    "- Se envolver Alares, Claro, Vivo, Tim, Oi ou provedores similares: use Internet e Telecom.\n"
    "- Se for pagamento de fatura de cartão de crédito: use Fatura Cartão.\n"
    "Classifique TODOS os lançamentos, sem omitir nenhum."
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


def format_brl(amount: float) -> str:
    """Formata um número float como moeda brasileira (R$ 1.234,56)."""
    if amount is None or pd.isna(amount):
        return "R$ 0,00"
    s = f"{float(amount):,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"


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


def extract_text_from_pdf_with_pymupdf(file_bytes: bytes) -> str:
    """Extrai texto bruto de um PDF usando PyMuPDF (fitz)."""
    texts: List[str] = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            t = page.get_text("text")
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
                "Este é um extrato bancário brasileiro. Extraia todos os lançamentos "
                "e retorne JSON com array de objetos contendo data (formato DD/MM/YYYY), "
                "descricao e valor (número positivo para crédito, negativo para débito)."
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


def extrair_lancamentos(arquivo) -> pd.DataFrame:
    """
    Detecta automaticamente se o arquivo é PDF ou CSV e retorna sempre
    um DataFrame com exatamente 3 colunas: data, descricao, valor.

    - CSV: testa separadores vírgula e ponto e vírgula, detecta automaticamente
      colunas de data, valor e descrição independente do nome ou ordem.
    - PDF com texto: usa pdfplumber; se extrair menos de 100 caracteres,
      tenta PyMuPDF. O texto extraído é enviado para a OpenAI para extrair
      os lançamentos em JSON.
    - PDF com imagem: converte com pdf2image e envia para OpenAI Vision gpt-4o
      para extrair lançamentos em JSON.
    """
    if arquivo is None:
        raise ValueError("Nenhum arquivo enviado.")

    name = (getattr(arquivo, "name", "") or "").lower()
    file_bytes = arquivo.read()
    if not file_bytes:
        raise ValueError("Arquivo vazio.")

    # -----------------
    # Arquivos CSV
    # -----------------
    if name.endswith(".csv"):
        # 1) Tenta ler com pandas usando diferentes separadores (vírgula e ponto e vírgula)
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
            # 2) Fallback: envia o conteúdo bruto do CSV para a OpenAI extrair os lançamentos
            csv_text = ""
            for encoding in ("utf-8", "latin-1"):
                try:
                    csv_text = file_bytes.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            if not csv_text:
                raise ValueError(
                    "Não foi possível decodificar o CSV nem extrair as colunas automaticamente."
                )

            df_out = extract_transactions_with_openai_from_text(csv_text)

    # -----------------
    # Arquivos PDF
    # -----------------
    elif name.endswith(".pdf"):
        # 1) Tenta extrair texto com pdfplumber
        try:
            raw_text = extract_text_from_pdf_with_pdfplumber(file_bytes)
        except Exception:
            raw_text = ""

        text_ok = raw_text.strip() if raw_text else ""

        # 2) Se texto insuficiente, tenta PyMuPDF
        if len(text_ok) < 100:
            try:
                raw_text_pymupdf = extract_text_from_pdf_with_pymupdf(file_bytes)
            except Exception:
                raw_text_pymupdf = ""

            if len((raw_text_pymupdf or "").strip()) >= 100:
                text_ok = raw_text_pymupdf.strip()

        if text_ok and len(text_ok) >= 100:
            # Texto suficiente: usa OpenAI para extrair lançamentos do texto
            df_out = extract_transactions_with_openai_from_text(text_ok)
        else:
            # 3) Possível PDF baseado em imagem: usa pdf2image + OpenAI Vision
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

            df_out = extract_transactions_with_openai_from_images(images)
    else:
        raise ValueError("Tipo de arquivo não suportado. Use PDF ou CSV.")

    if df_out is None or df_out.empty:
        raise ValueError("Nenhum lançamento foi identificado no arquivo enviado.")

    # Garante que o resultado tenha exatamente as 3 colunas esperadas,
    # independente da origem (CSV ou PDF)
    if not {"data", "descricao", "valor"}.issubset(df_out.columns):
        raise ValueError(
            "Os lançamentos extraídos não possuem as colunas esperadas: "
            "data, descricao, valor."
        )

    df_final = df_out[["data", "descricao", "valor"]].copy()
    return df_final.reset_index(drop=True)


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

    total = len(df)
    batch_size = 20

    progress_bar = st.progress(0)
    status_text = st.empty()

    result_rows: List[Dict] = []

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_df = df.iloc[start:end]

        status_text.text(
            f"Classificando lançamentos {start + 1}-{end} de {total}..."
        )
        progress_bar.progress(int((end / total) * 100))

        lancamentos_batch = []
        for _, row in batch_df.iterrows():
            lancamentos_batch.append(
                {
                    "data": row["data"],
                    "descricao": row["descricao"],
                    "valor": float(row["valor"])
                    if row["valor"] is not None
                    else 0.0,
                }
            )

        user_prompt = (
            "Classifique os lançamentos bancários a seguir conforme o prompt de sistema.\n"
            "Retorne APENAS JSON válido, sem texto extra nem markdown, no formato:\n"
            "[{"
            "data, descricao, valor, categoria, "
            "confianca: alta|media|baixa, motivo: explicação curta"
            "}]\n\n"
            "Processa todos os lançamentos sem omitir nenhum.\n\n"
            "Lançamentos em JSON:\n"
            f"{json.dumps(lancamentos_batch, ensure_ascii=False)}"
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
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError(
                "A resposta da OpenAI não pôde ser interpretada como JSON válido."
            )

        # Aceita tanto um array direto quanto um objeto com chave 'lancamentos'
        if isinstance(data, dict):
            lancamentos_resp = data.get("lancamentos")
        else:
            lancamentos_resp = data

        if not isinstance(lancamentos_resp, list):
            raise ValueError(
                "O JSON retornado pela OpenAI não contém uma lista de lançamentos válida."
            )

        for item in lancamentos_resp:
            categoria = str(item.get("categoria", "")).strip()
            confianca_raw = str(item.get("confianca", "")).strip().lower()
            motivo = str(item.get("motivo", "")).strip()

            # Normaliza confiança para Alta / Média / Baixa
            if "alta" in confianca_raw:
                confianca = "Alta"
            elif "med" in confianca_raw or "méd" in confianca_raw:
                confianca = "Média"
            elif "baix" in confianca_raw:
                confianca = "Baixa"
            else:
                confianca = "Baixa"

            result_rows.append(
                {
                    "data": item.get("data"),
                    "descricao": item.get("descricao"),
                    "valor": item.get("valor"),
                    "categoria": categoria,
                    "confianca": confianca,
                    "motivo": motivo,
                }
            )

    progress_bar.progress(100)
    status_text.empty()

    result_df = pd.DataFrame(result_rows)
    return result_df


# -----------------------------
# Funções de estilo, resumo e visual
# -----------------------------
def highlight_category(row: pd.Series):
    """
    Define a cor de fundo da linha de acordo com a categoria:
    - Verde: Receita
    - Vermelho: Despesa (não Receita, não Transferência, não Revisar)
    - Amarelo: Revisar
    """
    categoria = str(row.get("categoria", "")).lower()
    if "receita" in categoria:
        color = "#d4edda"  # verde claro
    elif "revisar" in categoria:
        color = "#fff3cd"  # amarelo claro
    elif "transfer" in categoria:
        color = ""  # neutro para transferências
    else:
        color = "#f8d7da"  # vermelho claro para demais despesas
    return [f"background-color: {color}" if color else "" for _ in row]


def compute_financial_summary(df: pd.DataFrame):
    """Calcula totais de lançamentos, receitas, despesas e saldo."""
    if df is None or df.empty:
        return 0, 0.0, 0.0, 0.0

    total_lanc = len(df)
    receitas = df.loc[df["valor"] > 0, "valor"].sum()
    despesas = df.loc[df["valor"] < 0, "valor"].sum()
    saldo = df["valor"].sum()
    return total_lanc, float(receitas), float(despesas), float(saldo)


def export_to_excel(df: pd.DataFrame) -> bytes:
    """Gera um arquivo Excel em memória a partir do DataFrame."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Lançamentos")
    return output.getvalue()


def plot_category_pie(df: pd.DataFrame):
    """Plota gráfico de pizza com distribuição por categoria."""
    if df.empty or "categoria" not in df.columns or "valor" not in df.columns:
        return

    summary = (
        df.assign(valor_abs=df["valor"].abs())
        .groupby("categoria", dropna=False)["valor_abs"]
        .sum()
        .reset_index()
    )

    fig = px.pie(
        summary,
        names="categoria",
        values="valor_abs",
        title="Distribuição por categoria",
    )
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

    # Header principal
    st.markdown(
        """
        <div style="text-align: left; margin-bottom: 1rem;">
            <h1 style="margin-bottom: 0;">📊 ConciBot</h1>
            <p style="margin-top: 0; color: #555;">
                Classificação inteligente de extratos bancários
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("Faça upload do seu extrato bancário em PDF ou CSV.")

    # Placeholder para barra de progresso global
    progress_placeholder = st.empty()

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
            progress = progress_placeholder.progress(10)
            with st.spinner("Lendo e analisando o extrato..."):
                df = extrair_lancamentos(uploaded_file)
                progress.progress(40)
            st.session_state.raw_df = df
            st.session_state.classified_df = None
            st.session_state.edited_df = None

            st.subheader("Lançamentos extraídos")
            st.dataframe(df, use_container_width=True)
            progress.progress(50)
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")
            progress_placeholder.empty()
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

        # Resumo rápido antes da tabela
        total_lanc, total_receitas, total_despesas, saldo = compute_financial_summary(
            current_df
        )
        st.markdown(f"**Total de lançamentos encontrados:** {total_lanc}")
        col_r, col_d = st.columns(2)
        with col_r:
            st.markdown(
                f"**Total de receitas:** "
                f"<span style='color: green;'>{format_brl(total_receitas)}</span>",
                unsafe_allow_html=True,
            )
        with col_d:
            st.markdown(
                f"**Total de despesas:** "
                f"<span style='color: red;'>{format_brl(abs(total_despesas))}</span>",
                unsafe_allow_html=True,
            )

        # Tabela colorida por confiança (visual)
        display_df = current_df[["data", "descricao", "valor", "categoria"]]
        styled = display_df.style.apply(highlight_category, axis=1).format(
            {"valor": format_brl}
        )
        st.markdown("**Tabela colorida por categoria:**")
        try:
            # Exibe tabela estática com cores
            st.table(styled)
        except Exception:
            # Se por algum motivo o Styler não funcionar, cai para a tabela simples
            st.dataframe(display_df, use_container_width=True)

        # Editor interativo para permitir edição de categorias
        st.markdown("**Edite as categorias diretamente na tabela abaixo, se necessário:**")
        edited_df = st.data_editor(
            current_df,
            hide_index=True,
            use_container_width=True,
            column_order=["data", "descricao", "valor", "categoria"],
            column_config={
                "data": st.column_config.TextColumn("Data", disabled=True),
                "descricao": st.column_config.TextColumn("Descrição", disabled=True),
                "valor": st.column_config.NumberColumn(
                    "Valor",
                    disabled=True,
                    format="R$ %.2f",
                ),
                "categoria": st.column_config.TextColumn("Categoria"),
            },
            key="editor",
        )
        st.session_state.edited_df = edited_df

        # ---------------------------------
        # Resumo final: cards e gráfico
        # ---------------------------------
        if st.session_state.edited_df is not None:
            total_lanc, total_receitas, total_despesas, saldo = (
                compute_financial_summary(st.session_state.edited_df)
            )

            st.markdown("### Resumo do período")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total de lançamentos", f"{total_lanc}")
            c2.metric("Total de receitas", format_brl(total_receitas))
            c3.metric("Total de despesas", format_brl(abs(total_despesas)))
            c4.metric("Saldo do período", format_brl(saldo))

            st.markdown("### Exportação e gráficos")
            col1, col2 = st.columns([1, 1])

            with col1:
                excel_bytes = export_to_excel(st.session_state.edited_df)
                st.download_button(
                    "Exportar Excel",
                    data=excel_bytes,
                    file_name="concibot_lancamentos_classificados.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    type="primary",
                )

            with col2:
                st.markdown("**Distribuição por categoria:**")
                plot_category_pie(st.session_state.edited_df)


if __name__ == "__main__":
    main()

