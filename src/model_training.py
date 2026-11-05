import streamlit as st                                # Framework para criação da interface web
from sklearn.model_selection import train_test_split  # Divide os dados em treino e teste
from sklearn.preprocessing import OrdinalEncoder      # Converte categorias de texto em números
from sklearn.naive_bayes import CategoricalNB         # Naive Bayes para dados categórico
from imblearn.over_sampling import RandomOverSampler

# ─── FUNÇÃO: TREINAR O MODELO ────────────────────────────────────────────────

@st.cache_resource
def model(df):
    """
    Treina o modelo Naive Bayes com balanceamento de classes (Over-sampling) no treino,
    garantindo divisão estratificada e gera as previsões do conjunto de teste.

    Parâmetros:
    df (pd.DataFrame): dados originais retornados por load_data()

    Retorna:
    tuple: (encoder, modelo, y_test, y_pred), nesta ordem
    """

    # Converte as colunas de entrada para o tipo category
    for col in df.columns.drop('class'):
        df[col] = df[col].astype('category')

    # Inicializa o OrdinalEncoder — converte categorias de texto em números
    encoder = OrdinalEncoder()

    # Aplica o encoder nas features (colunas de entrada)
    X_encoded = encoder.fit_transform(df.drop('class', axis=1))

    # Converte a coluna alvo 'class' em códigos numéricos
    y = df['class'].astype('category').cat.codes

    # 1. Divide os dados em treino (70%) e teste (30%) COM ESTRATIFICAÇÃO
    # stratify=y garante que todas as classes mantenham a proporção correta
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, random_state=42, test_size=0.3, stratify=y
    )

    # 2. Aplica o RandomOverSampler APENAS no conjunto de treino
    # Isso resolve o problema de 0% de recall na classe minoritária ("Muito Bom")
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    # Inicializa o modelo Naive Bayes para dados categóricos
    modelo = CategoricalNB()

    # Treina o modelo usando os dados balanceados de treino
    modelo.fit(X_train_resampled, y_train_resampled)

    # Faz previsões no conjunto de teste original (não balanceado)
    y_pred = modelo.predict(X_test)

    # Retorna o encoder, o modelo treinado, os valores reais e os previstos    
    return encoder, modelo, y_test, y_pred