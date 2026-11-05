import streamlit as st
from src.data_loader import load_data
from src.model_training import model
from src.charts import metricas, bar_chart

st.set_page_config(
    page_title="Visão Geral - Classificação de Veículos",
    page_icon="🚗",
    layout="wide"
)

st.title('Visão Geral')

# ─── LEITURA DOS DADOS E TREINO DO MODELO ────────────────────────────────────

# Lê o CSV original (colunas e valores em inglês)
df = load_data()

# Desempacota as variáveis do model
encoder, modelo, y_test, y_pred = model(df)

# ─── TÍTULO ──────────────────────────────────────────────────────────────────

st.title('🚗 Classificação de Veículos com Naive Bayes')
st.caption('Utilizando o Conjunto de dados de avaliação de automóveis')

# Calcula as 4 métricas gerais do modelo (função de charts.py)
acuracia, precisao, recall, f1 = metricas(y_test, y_pred)

# ─── DICIONÁRIOS DE TRADUÇÃO ─────────────────────────────────────────────────
# Servem para exibir os dados em português sem alterar os dados usados no treino

traducao = {
    # Preço de compra
    'buying': {
        'vhigh': 'Muito Alto', 'high': 'Alto',
        'med':   'Médio',      'low':  'Baixo'
    },
    # Custo de manutenção
    'maint': {
        'vhigh': 'Muito Alta', 'high': 'Alta',
        'med':   'Média',      'low':  'Baixa'
    },
    # Número de portas
    'doors': {
        '2': '2 Portas',  '3': '3 Portas',
        '4': '4 Portas',  '5more': '5 ou mais'
    },
    # Capacidade de passageiros
    'persons': {
        '2': '2 Pessoas', '4': '4 Pessoas',
        'more': 'Mais de 4'
    },
    # Tamanho do porta-malas
    'lug_boot': {
        'small': 'Pequeno', 'med': 'Médio',
        'big':   'Grande'
    },
    # Nível de segurança
    'safety': {
        'low': 'Baixa', 'med': 'Média',
        'high': 'Alta'
    },
    # Classe (resultado) — usada para traduzir a coluna 'class' da tabela
    'class': {
        'unacc': 'Inaceitável',
        'acc':   'Aceitável',
        'good':  'Bom',
        'vgood': 'Muito Bom'
    }
}

# Tradução reversa: português → inglês
# {v: k for k, v in mapa.items()} → inverte chave e valor de cada dicionário
# Necessária porque o encoder foi treinado com os valores em inglês
traducao_reversa = {
    col: {v: k for k, v in mapa.items()}
    for col, mapa in traducao.items()
}

# ─── DATAFRAME TRADUZIDO (SOMENTE PARA EXIBIÇÃO) ─────────────────────────────

# Cópia do DataFrame: o original continua em inglês
df_traduzido = df.copy()

# Percorre cada coluna do dicionário de tradução
for key, value in traducao.items():
    if key in df_traduzido.columns:
        # astype(str) → garante texto | map(value) → traduz pelo dicionário
        # fillna(...) → mantém o valor original se não houver tradução
        df_traduzido[key] = df_traduzido[key].astype(str).map(value).fillna(df_traduzido[key])

# Nomes das colunas em português
traducao_colunas = {
    'buying':   'Preço de Compra',
    'maint':    'Manutenção',
    'doors':    'Portas',
    'persons':  'Capacidade',
    'lug_boot': 'Porta-Malas',
    'safety':   'Segurança',
    'class':    'Classe'
}

df_traduzido = df_traduzido.rename(columns=traducao_colunas)

# ─── KPIs ────────────────────────────────────────────────────────────────────

m1, m2, m3, m4 = st.columns(4)

with m1:
    with st.container(border=True):
        st.metric('📄 Total de Registros', f'{len(df):,}'.replace(',', '.')) # Total de linhas
with m2:
    with st.container(border=True):
        st.metric('📊 Atributos Analisados', len(df.columns.drop("class"))) # Colunas, sem a classe
with m3:
    with st.container(border=True):
        st.metric('🏷️ Classes de Veículos', df['class'].nunique()) # Classes distintas
with m4:
    with st.container(border=True):
        st.metric('Ausentes', int(df.isna().sum().sum())) # Total de valores nulos


col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader('📊 Distribuição das Classes')
        st.caption('Gráfico de barras com a quantidade de registros por classe')
        bar_chart(df_traduzido)

with col2:
    with st.container(border=True):
        st.subheader('📋 Amostra dos Dados')
        st.caption('Tabela com os 10 primeiros registros do dataset, traduzidos para português')
        st.dataframe(
            df_traduzido.head(10), 
            width='stretch',
            hide_index=True,
            height=380
        )


# Rodapé explicativo contextual
st.info("💡 **Dica de Navegação:** Utilize a barra lateral esquerda para alternar entre as páginas de Exploração de Dados, Desempenho do Modelo, Previsões e Relatórios Técnicos.")


