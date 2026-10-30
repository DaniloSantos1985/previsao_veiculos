# ─── IMPORTAÇÕES ─────────────────────────────────────────────────────────────

import pandas as pd                                   # Manipulação e leitura de dados
import streamlit as st                                # Framework para criação da interface web
from sklearn.model_selection import train_test_split  # Divide os dados em treino e teste
from sklearn.preprocessing import OrdinalEncoder      # Converte categorias de texto em números
from sklearn.naive_bayes import CategoricalNB         # Naive Bayes para dados categóricos


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


# ─── FUNÇÃO: TREINAR O MODELO ────────────────────────────────────────────────

# Também usa cache: o modelo só é treinado uma vez, não a cada clique
@st.cache_resource
def model(df):
    """
    Treina o modelo Naive Bayes e gera as previsões do conjunto de teste.

    Parâmetros:
    df (pd.DataFrame): dados originais retornados por load_data()

    Retorna:
    tuple: (encoder, modelo, y_test, y_pred), nesta ordem
    """

    # Converte as colunas de entrada (todas, exceto 'class') para o tipo category
    # Obs.: o OrdinalEncoder aceita texto direto, então este passo não é obrigatório
    for col in df.columns.drop('class'):
        df[col] = df[col].astype('category')

    # Inicializa o OrdinalEncoder — converte categorias de texto em números
    # As categorias de cada coluna são ordenadas em ordem ALFABÉTICA
    # Exemplo na coluna 'buying': high → 0, low → 1, med → 2, vhigh → 3
    # O encoder é treinado com os valores ORIGINAIS em inglês, por isso a
    # previsão precisa reverter a tradução antes de chamar encoder.transform()
    encoder = OrdinalEncoder()

    # Aplica o encoder nas features (colunas de entrada)
    # drop('class', axis=1) → remove a coluna alvo antes de codificar
    # fit_transform → aprende as categorias de cada coluna e já transforma
    X_encoded = encoder.fit_transform(df.drop('class', axis=1))

    # Converte a coluna alvo 'class' em códigos numéricos
    # astype('category').cat.codes → acc=0, good=1, unacc=2, vgood=3
    y = df['class'].astype('category').cat.codes

    # Inicializa o modelo Naive Bayes para dados categóricos
    modelo = CategoricalNB()

    # Divide os dados em treino (70%) e teste (30%)
    # random_state=42 → garante a mesma divisão a cada execução
    # test_size=0.3   → 30% dos registros reservados para teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, random_state=42, test_size=0.3
    )

    # Treina o modelo apenas com os dados de treino
    modelo.fit(X_train, y_train)

    # Faz previsões no conjunto de teste para avaliar o modelo depois
    y_pred = modelo.predict(X_test)

    # Retorna o encoder, o modelo treinado, os valores reais e os previstos
    # ⚠️ A ordem é y_test e depois y_pred
    return encoder, modelo, y_test, y_pred