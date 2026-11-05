import streamlit as st
from src.data_loader import load_data

st.set_page_config(
    page_title="Exploração de Dados - Classificaçãod de Veículos",
    layout="wide",
    page_icon="🔍"
)

df = load_data()

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

st.title('Exploração e Análise de Dados')
st.markdown("Ferramenta interativa de exploração do dataset, permitindo filtros avançados, " \
"pesquisa textual e exportação.")

st.sidebar.header('⚙️ Painel de Filtros')


with st.sidebar.expander("🔍 Filtros Avançados", expanded=True):
    st.markdown("Use o filtro abaixo para explorar o dataset de forma interativa. " \
    "Você pode selecionar múltiplos valores para cada classe e visualizar os resultados em tempo real.")

    classes_selecionadas = st.multiselect(
        'Selecione as classes dos veículos:',
        options = df_traduzido['Classe'].unique(),
        default = df_traduzido['Classe'].unique()
    )

df_filtrado = df_traduzido[df_traduzido['Classe'].isin(classes_selecionadas)]

total_registros = len(df_traduzido)
registros_exibidos = len(df_filtrado)

m1, m2 = st.columns(2)

with m1:
    with st.container(border=True):
        st.metric(
            '📄 Total de Registros', 
            f'{total_registros:,}'.replace(',', '.'),
            f'{total_registros / total_registros:.0%}'
        ) # Total de linhas

with m2:
    with st.container(border=True):
        st.metric(
            '📊 Registros Selecionados', 
            f'{registros_exibidos:,}'.replace(',', '.'),
            f'{registros_exibidos / total_registros:.0%}'
        ) # Total de linhas selecionadas

with st.container(border=True):

    st.subheader("📋 Amostra e Detalhamento dos Dados")
    st.markdown(f"Exibindo **{registros_exibidos}** de **{total_registros}** registros disponíveis.")


    st.dataframe(
        df_filtrado,
        width='stretch',
        hide_index=True,
        height=450
    )

    col_esquerda, col_direita = st.columns([6, 1])

    with col_direita:        
        csv = df_filtrado.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Baixar como CSV",
            data=csv,
            file_name='dados_filtrados.csv',
            mime='text/csv',
            width='stretch'
        )

with st.container(border=True):
    st.subheader("📊 Estatísticas e Frequências por Categoria")

    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown('Frequencia absoluta por Classes')
        df_frequencia = df_filtrado['Classe'].value_counts().reset_index()
        df_frequencia = df_frequencia.rename(
            columns={
                'Classe':'Classes',
                'count':'Quantidade'
            }
        )
        st.dataframe(
            df_frequencia,
            width='stretch',
            hide_index=True,
            height=170
        )

    with col_b:
        st.markdown("**Resumo Estatístico dos Atributos**")
        df_resumo = (
            df_filtrado.describe(include='all')
            .T        
            .rename(columns={            
                'count':'Quantidade', 
                'unique':'Únicos', 
                'top':'Mais Frequente', 
                'freq':'Frequência'            
            })
        )
        st.dataframe(df_resumo, width='stretch')