# ═════════════════════════════════════════════════════════════════════════════
# 01_Visão_Geral.py — Página de visão geral do dataset
#
# Função: apresentar os indicadores básicos do dataset, a distribuição
# das classes e uma amostra dos dados traduzidos para português.
# ═════════════════════════════════════════════════════════════════════════════

# ─── IMPORTAÇÕES ─────────────────────────────────────────────────────────────

import streamlit as st                          # Framework para criação da interface web
from src.data_loader import load_data           # Lê o CSV de veículos
from src.model_training import model            # Treina o modelo e devolve encoder, modelo e previsões de teste
from src.charts import metricas, bar_chart      # metricas → calcula acurácia, precisão, recall e F1
                                                # bar_chart → gráfico de barras da distribuição das classes

# ─── CONFIGURAÇÃO DA PÁGINA ──────────────────────────────────────────────────

# Título da aba, ícone e layout em tela cheia
# Deve ser o primeiro comando Streamlit do arquivo
st.set_page_config(
    page_title="Visão Geral - Classificação de Veículos",
    page_icon="🚗",
    layout="wide"
)

st.title('Visão Geral')

# ─── LEITURA DOS DADOS E TREINO DO MODELO ────────────────────────────────────

# Lê o CSV original (colunas e valores em inglês)
df = load_data()

# model() devolve: encoder, modelo, y_test, y_pred (nessa ordem)
# O treino usa cache, então só acontece na primeira execução
encoder, modelo, y_test, y_pred = model(df)

# ─── TÍTULO ──────────────────────────────────────────────────────────────────

# ⚠️ Esta é a segunda chamada de st.title na página; a primeira está acima ('Visão Geral')
st.title('🚗 Classificação de Veículos com Naive Bayes')
st.caption('Utilizando o Conjunto de dados de avaliação de automóveis')

# Calcula as 4 métricas gerais do modelo (função de charts.py)
# ⚠️ Os valores são calculados mas ainda não são exibidos nesta página
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
# Necessária na página de previsões, porque o encoder foi treinado com valores em inglês
# (não é usada nesta página)
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
        # astype(str) → garante texto
        # map(value)  → traduz cada valor pelo dicionário
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

# Quatro colunas de largura igual, uma para cada indicador
m1, m2, m3, m4 = st.columns(4)

# Cada métrica fica dentro de um container com borda (cartão)
with m1:
    with st.container(border=True):
        # f'{len(df):,}' → separador de milhar com vírgula (1,000)
        # .replace(',', '.') → converte para o padrão brasileiro (1.000)
        st.metric('📄 Total de Registros', f'{len(df):,}'.replace(',', '.'))
with m2:
    with st.container(border=True):
        # df.columns.drop("class") → todas as colunas, menos a classe (variável alvo)
        st.metric('📊 Atributos Analisados', len(df.columns.drop("class")))
with m3:
    with st.container(border=True):
        # nunique() → quantidade de classes distintas
        st.metric('🏷️ Classes de Veículos', df['class'].nunique())
with m4:
    with st.container(border=True):
        # isna().sum().sum() → total de valores nulos em todo o DataFrame
        st.metric('Ausentes', int(df.isna().sum().sum()))

# ─── GRÁFICO E TABELA ────────────────────────────────────────────────────────

# Duas colunas de largura igual: gráfico à esquerda, tabela à direita
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader('📊 Distribuição das Classes')
        st.caption('Gráfico de barras com a quantidade de registros por classe')
        # Usa o DataFrame traduzido: a função espera a coluna 'Classe'
        bar_chart(df_traduzido)

with col2:
    with st.container(border=True):
        st.subheader('📋 Amostra dos Dados')
        st.caption('Tabela com os 10 primeiros registros do dataset, traduzidos para português')
        # head(10)         → primeiras 10 linhas
        # width='stretch'  → ocupa toda a largura da coluna
        # hide_index=True  → esconde a coluna de índice numérico
        # height=380       → mesma altura do gráfico, para os dois ficarem alinhados
        st.dataframe(
            df_traduzido.head(10),
            width='stretch',
            hide_index=True,
            height=380
        )

# ─── DICA DE NAVEGAÇÃO ───────────────────────────────────────────────────────

# Caixa informativa azul no rodapé da página
st.info("💡 **Dica de Navegação:** Utilize a barra lateral esquerda para alternar entre as páginas de Exploração de Dados, Desempenho do Modelo, Previsões e Relatórios Técnicos.")