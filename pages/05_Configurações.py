# ═════════════════════════════════════════════════════════════════════════════
# 05_Configurações.py — Página de configurações e metadados do sistema
#
# Função: mostrar informações do modelo, as versões das bibliotecas do
# ambiente e os parâmetros usados no pré-processamento e no treino.
# ═════════════════════════════════════════════════════════════════════════════

# ─── IMPORTAÇÕES ─────────────────────────────────────────────────────────────

import streamlit as st                          # Framework para criação da interface web
import sys                                      # Informações do interpretador Python (versão)
import sklearn                                  # Usado para ler a versão do scikit-learn
import pandas as pd                             # Manipulação de dados e leitura da versão do pandas
from src.data_loader import load_data           # Lê o CSV de veículos
from src.model_training import model            # Treina o modelo e devolve encoder, modelo e previsões de teste

# ─── CONFIGURAÇÃO DA PÁGINA ──────────────────────────────────────────────────

# Título da aba, ícone e layout em tela cheia
# Deve ser o primeiro comando Streamlit do arquivo
st.set_page_config(
    page_title="Configurações e Metadados - Classificação de Veículos",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Configurações e Parâmetros do Sistema")
st.markdown("Informações detalhadas sobre o ambiente de execução, versões das bibliotecas, metadados do modelo e parâmetros globais.")

# ─── DADOS E MODELO ──────────────────────────────────────────────────────────

# Carrega os dados e obtém o modelo treinado
# model() devolve: encoder, modelo, y_test, y_pred (nessa ordem)
df = load_data()
encoder, modelo, y_test, y_pred = model(df)

# ==========================================
# SEÇÃO 1: METADADOS DO MODELO DE MACHINE LEARNING
# ==========================================
st.subheader("🤖 Metadados do Modelo Ativo")

# Três indicadores lado a lado
col1, col2, col3 = st.columns(3)

# Os três valores abaixo são texto escrito à mão: não acompanham mudanças no código de treino
with col1:
    st.metric(label="Algoritmo Principal", value="Categorical Naive Bayes")
with col2:
    st.metric(label="Estratégia de Balanceamento", value="RandomOverSampler")
with col3:
    # "Produção" é um rótulo forte para um projeto educacional
    st.metric(label="Versão da Arquitetura", value="v1.0 (Produção)")

# Linha divisória horizontal
st.markdown("---")

# ==========================================
# SEÇÃO 2: ESPECIFICAÇÕES DO AMBIENTE E STACK
# ==========================================
st.subheader("🛠️ Stack Tecnológica e Versões do Ambiente")

# Organiza as versões das principais bibliotecas utilizadas no projeto
# Dicionário no formato {coluna: lista de valores}, que o pandas converte em tabela
env_data = {
    "Componente": [
        "Python",
        "Streamlit",
        "Scikit-Learn",
        "Pandas",
        "NumPy"
    ],
    "Versão Ativa": [
        # sys.version_info → versão do Python em uso: major.minor.micro (ex.: 3.12.4)
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        st.__version__,        # Versão instalada do Streamlit
        sklearn.__version__,   # Versão instalada do scikit-learn
        pd.__version__,        # Versão instalada do pandas        
        # e esta linha sempre mostra "Compatível" em vez da versão do NumPy
        # Correção: import numpy as np e usar np.__version__
        pd.np.__version__ if hasattr(pd, 'np') else "Compatível"
    ]
}

# Converte o dicionário em DataFrame para exibir como tabela
df_env = pd.DataFrame(env_data)

# Exibe a tabela com as versões das bibliotecas e do Python
st.dataframe(
    df_env,
    width='stretch',
    hide_index=True
)

st.markdown("---")

# ==========================================
# SEÇÃO 3: PARÂMETROS DE EXECUÇÃO
# ==========================================
st.subheader("📋 Parâmetros e Hiperparâmetros Globais")

# expander → seção recolhível, fechada por padrão
with st.expander("🔍 Ver configurações de pré-processamento e treino"):    
    # (divisão 70/30, stratify=y, random_state=42 e o oversampling só no treino)
    st.markdown("""
    * **Divisão de Dados (Train/Test Split):** 70% para treino e 30% para teste.
    * **Estratificação:** Ativada (`stratify=y`) para preservação da proporção das classes originais.
    * **Codificação de Variáveis:** Conversão ordinal via `OrdinalEncoder` para suporte nativo ao classificador categórico.
    * **Reprodutibilidade:** Fixada com `random_state=42` em todas as etapas estocásticas (divisão de dados e oversampling).
    """)

# Painel de controle do sistema, com informações sobre o dashboard corporativo
st.caption("💡 *Painel de controle gerado para monitoramento e auditoria do dashboard corporativo.*")