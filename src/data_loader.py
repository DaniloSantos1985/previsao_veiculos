# ─── IMPORTAÇÕES ─────────────────────────────────────────────────────────────
import pandas as pd                                   # Manipulação e leitura de dados
import streamlit as st                                # Framework para criação da interface web
# ─── FUNÇÃO: CARREGAR DADOS ──────────────────────────────────────────────────

# @st.cache_data → guarda o resultado em cache após a primeira execução
# Evita reler o CSV a cada interação do usuário com a página
@st.cache_data
def load_data():
    """
    Carrega o dataset de veículos a partir do CSV.

    Retorna:
    pd.DataFrame: dados originais, com colunas em inglês
                  (buying, maint, doors, persons, lug_boot, safety, class)
    """

    # Lê o arquivo CSV com os dados de veículos
    df = pd.read_csv('data/car.csv', sep=',')

    return df