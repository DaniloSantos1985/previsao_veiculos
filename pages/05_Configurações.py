import streamlit as st
import sys
import sklearn
import pandas as pd
from src.data_loader import load_data
from src.model_training import model

# Configuração da página
st.set_page_config(
    page_title="Configurações e Metadados - Classificação de Veículos",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Configurações e Parâmetros do Sistema")
st.markdown("Informações detalhadas sobre o ambiente de execução, versões das bibliotecas, metadados do modelo e parâmetros globais.")

# Carrega os dados e obtém o modelo treinado
df = load_data()
encoder, modelo, y_test, y_pred = model(df)

# ==========================================
# SEÇÃO 1: METADADOS DO MODELO DE MACHINE LEARNING
# ==========================================
st.subheader("🤖 Metadados do Modelo Ativo")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Algoritmo Principal", value="Categorical Naive Bayes")
with col2:
    st.metric(label="Estratégia de Balanceamento", value="RandomOverSampler")
with col3:
    st.metric(label="Versão da Arquitetura", value="v1.0 (Produção)")

st.markdown("---")

# ==========================================
# SEÇÃO 2: ESPECIFICAÇÕES DO AMBIENTE E STACK
# ==========================================
st.subheader("🛠️ Stack Tecnológica e Versões do Ambiente")

# Organiza as versões das principais bibliotecas utilizadas no projeto
env_data = {
    "Componente": [
        "Python",
        "Streamlit",
        "Scikit-Learn",
        "Pandas",
        "NumPy"
    ],
    "Versão Ativa": [
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        st.__version__,
        sklearn.__version__,
        pd.__version__,
        pd.np.__version__ if hasattr(pd, 'np') else "Compatível"
    ]
}

df_env = pd.DataFrame(env_data)

st.dataframe(
    df_env,
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# ==========================================
# SEÇÃO 3: PARÂMETROS DE EXECUÇÃO
# ==========================================
st.subheader("📋 Parâmetros e Hiperparâmetros Globais")

with st.expander("🔍 Ver configurações de pré-processamento e treino"):
    st.markdown("""
    * **Divisão de Dados (Train/Test Split):** 70% para treino e 30% para teste.
    * **Estratificação:** Ativada (`stratify=y`) para preservação da proporção das classes originais.
    * **Codificação de Variáveis:** Conversão ordinal via `OrdinalEncoder` para suporte nativo ao classificador categórico.
    * **Reprodutibilidade:** Fixada com `random_state=42` em todas as etapas estocásticas (divisão de dados e oversampling).
    """)

st.caption("💡 *Painel de controle gerado dinamicamente para monitoramento e auditoria do dashboard corporativo.*")