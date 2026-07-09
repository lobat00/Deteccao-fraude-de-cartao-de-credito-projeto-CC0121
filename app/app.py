"""
App Streamlit — Detecção de Fraude em Transações de Cartão de Crédito
CC0121 — Inteligência Artificial — UNIFAP — 2026.1

Como rodar:
    streamlit run app/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ---------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------
st.set_page_config(
    page_title="Detecção de Fraude — Cartão de Crédito",
    page_icon="💳",
    layout="centered",
)

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"

# Ordem EXATA usada no treino: o CSV original tem as colunas
# Time, V1, V2, ..., V28, Amount, Class — e removemos só Class.
FEATURE_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# ---------------------------------------------------------
# Carregar modelo e scaler (com cache para não recarregar a cada interação)
# ---------------------------------------------------------
@st.cache_resource
def carregar_artefatos():
    modelo = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return modelo, scaler


@st.cache_data
def carregar_dataset():
    # Usado só para o botão "carregar exemplo" — não é usado na predição
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


try:
    modelo, scaler = carregar_artefatos()
except FileNotFoundError:
    st.error(
        "Modelo ou scaler não encontrados. Rode antes os notebooks "
        "`02_preprocessing.ipynb` e `03_modeling.ipynb` para gerar "
        "`models/scaler.pkl` e `models/best_model.pkl`."
    )
    st.stop()

df_dataset = carregar_dataset()

# ---------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------
st.title("💳 Detecção de Fraude em Cartão de Crédito")
st.markdown(
    "Insira os atributos de uma transação para classificá-la como "
    "**legítima** ou **fraudulenta**, com o nível de confiança do modelo."
)
st.caption(
    "As colunas V1–V28 são componentes anonimizados via PCA "
    "(dataset Kaggle *Credit Card Fraud Detection*). "
    "Amount é o valor da transação e Time o tempo em segundos desde a primeira transação do dataset."
)

st.divider()

# ---------------------------------------------------------
# Botões para carregar exemplo real do dataset
# ---------------------------------------------------------


def carregar_exemplo(classe: int):
    """Sorteia uma linha do dataset com a classe indicada e preenche os campos do formulário."""
    if df_dataset is None:
        st.session_state["_erro_exemplo"] = (
            "Não encontrei data/creditcard.csv. Baixe o dataset para usar esta função."
        )
        return

    linha = df_dataset[df_dataset["Class"] == classe].sample(1).iloc[0]

    for col in FEATURE_COLS:
        st.session_state[col] = float(linha[col])

    st.session_state["_erro_exemplo"] = None


st.subheader("Carregar exemplo real do dataset (opcional)")
col_ex1, col_ex2 = st.columns(2)
with col_ex1:
    st.button(
        "🚨 Carregar exemplo de FRAUDE",
        use_container_width=True,
        on_click=carregar_exemplo,
        args=(1,),
    )
with col_ex2:
    st.button(
        "✅ Carregar exemplo LEGÍTIMO",
        use_container_width=True,
        on_click=carregar_exemplo,
        args=(0,),
    )

if st.session_state.get("_erro_exemplo"):
    st.warning(st.session_state["_erro_exemplo"])

st.divider()

# ---------------------------------------------------------
# Formulário de entrada
# ---------------------------------------------------------
with st.form("transacao_form"):
    st.subheader("Amount e Time")
    col_a, col_t = st.columns(2)
    with col_a:
        st.number_input(
            "Amount (valor da transação)", min_value=0.0, step=1.0, key="Amount"
        )
    with col_t:
        st.number_input(
            "Time (segundos desde a primeira transação)", min_value=0.0, step=1.0, key="Time"
        )

    st.subheader("Componentes V1–V28 (PCA)")
    st.caption("Valores tipicamente entre -30 e 30. Use os botões acima para preencher com um exemplo real.")

    cols = st.columns(4)
    for i in range(1, 29):
        col = cols[(i - 1) % 4]
        key = f"V{i}"
        col.number_input(key, value=0.0, format="%.4f", key=key)

    submitted = st.form_submit_button("Classificar Transação", use_container_width=True)

# ---------------------------------------------------------
# Predição
# ---------------------------------------------------------
if submitted:
    entrada = {col: st.session_state[col] for col in FEATURE_COLS}
    df_entrada = pd.DataFrame([entrada])[FEATURE_COLS]

    # Aplicar o mesmo scaler usado no treino (só em Amount e Time)
    df_entrada_scaled = df_entrada.copy()
    df_entrada_scaled[["Amount", "Time"]] = scaler.transform(df_entrada[["Amount", "Time"]])

    pred = modelo.predict(df_entrada_scaled)[0]
    proba = modelo.predict_proba(df_entrada_scaled)[0]
    confianca = proba[pred] * 100

    st.divider()
    st.subheader("Resultado")

    if pred == 1:
        st.error("🚨 **Transação classificada como FRAUDE**")
        st.metric("Confiança do modelo", f"{confianca:.2f}%")
        st.progress(int(proba[1] * 100))
    else:
        st.success("✅ **Transação classificada como LEGÍTIMA**")
        st.metric("Confiança do modelo", f"{confianca:.2f}%")
        st.progress(int(proba[0] * 100))

    with st.expander("Ver probabilidades detalhadas"):
        st.write(
            pd.DataFrame(
                {"Classe": ["Legítima", "Fraude"], "Probabilidade": [f"{proba[0]*100:.2f}%", f"{proba[1]*100:.2f}%"]}
            )
        )

st.divider()
st.caption(
    "Projeto Final — CC0121 (Inteligência Artificial) — UNIFAP — 2026.1 · "
    "Modelo: Random Forest treinado com SMOTE sobre o dataset Credit Card Fraud Detection (Kaggle)."
)
