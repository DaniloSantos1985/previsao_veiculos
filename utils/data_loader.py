# ─── IMPORTAÇÕES ─────────────────────────────────────────────────────────────

import pandas as pd                                  # Manipulação e leitura de dados
import streamlit as st                               # Framework para criação da interface web
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split # Divide os dados em treino e teste
from sklearn.preprocessing import OrdinalEncoder     # Converte categorias em números ordinais
from sklearn.naive_bayes import CategoricalNB        # Modelo Naive Bayes para dados categóricos

# ─── FUNÇÃO: CARREGAR DADOS E TREINAR MODELO ─────────────────────────────────

# @st.cache_data → armazena o resultado em cache após a primeira execução
# Evita recarregar o CSV e retreinar o modelo a cada interação do usuário
@st.cache_data
def load_data():
    """
    Carrega o dataset, treina o modelo Naive Bayes e retorna
    o DataFrame, o encoder, o modelo treinado e a acurácia.
    """

    # Lê o arquivo CSV com os dados de veículos
    df = pd.read_csv('data/car.csv', sep=',')

    return df

@st.cache_data
def model(df):
    # Converte todas as colunas (exceto 'class') para o tipo category
    # Necessário para o CategoricalNB aceitar os dados corretamente
    for col in df.columns.drop('class'):
        df[col] = df[col].astype('category')

    # Inicializa o OrdinalEncoder — converte categorias de texto em números
    # Exemplo: 'low' → 0, 'med' → 1, 'high' → 2, 'vhigh' → 3
    # IMPORTANTE: o encoder é treinado com os valores ORIGINAIS em inglês
    # Traduzir antes do treino alteraria a ordem ordinal e prejudicaria o modelo
    encoder = OrdinalEncoder()

    # Aplica o OrdinalEncoder nas features (colunas de entrada)
    # drop('class', axis=1) → remove a coluna alvo antes de encodar
    # fit_transform → aprende os valores únicos e transforma em números
    X_encoded = encoder.fit_transform(df.drop('class', axis=1))

    # Converte a coluna alvo 'class' em códigos numéricos
    # astype('category').cat.codes → 'acc'=0, 'good'=1, 'unacc'=2, 'vgood'=3
    y = df['class'].astype('category').cat.codes

    # Inicializa o modelo Naive Bayes para dados categóricos
    modelo = CategoricalNB()

    # Divide os dados em treino (70%) e teste (30%)
    # random_state=42 → garante reprodutibilidade da divisão
    # test_size=0.3   → 30% dos dados reservados para teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, random_state=42, test_size=0.3
    )

    # Treina o modelo com os dados de treino
    modelo.fit(X_train, y_train)

    # Faz previsões nos dados de teste para avaliar o modelo
    y_pred = modelo.predict(X_test)

    # Retorna o DataFrame original, o encoder treinado, o modelo e a acurácia
    return encoder, modelo, y_test, y_pred


