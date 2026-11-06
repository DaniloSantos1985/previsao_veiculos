# ═════════════════════════════════════════════════════════════════════════════
# 02_Exploração_de_Dados.py — Página de exploração interativa do dataset
#
# Função: permitir filtrar o dataset por classe, visualizar os registros,
# exportar o resultado em CSV e consultar frequências e estatísticas.
# ═════════════════════════════════════════════════════════════════════════════

# ─── IMPORTAÇÕES ─────────────────────────────────────────────────────────────

import streamlit as st                      # Framework para criação da interface web
from src.data_loader import load_data       # Lê o CSV de veículos

# ─── CONFIGURAÇÃO DA PÁGINA ──────────────────────────────────────────────────

# Título da aba, layout em tela cheia e ícone
st.set_page_config(
    page_title="Exploração de Dados - Classificaçãod de Veículos",
    layout="wide",
    page_icon="🔍"
)

# ─── LEITURA DOS DADOS ───────────────────────────────────────────────────────

# Lê o CSV original (colunas e valores em inglês)
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

# ─── CABEÇALHO ───────────────────────────────────────────────────────────────

st.title('Exploração e Análise de Dados')

# A barra invertida (\) no fim da linha permite quebrar a string em duas linhas no código
st.markdown("Ferramenta interativa de exploração do dataset, permitindo filtros avançados, " \
"pesquisa textual e exportação.")

# ─── SIDEBAR: FILTROS ────────────────────────────────────────────────────────

st.sidebar.header('⚙️ Painel de Filtros')

# expander → seção recolhível na barra lateral; expanded=True já abre por padrão
with st.sidebar.expander("🔍 Filtros Avançados", expanded=True):
    st.markdown("Use o filtro abaixo para explorar o dataset de forma interativa. " \
    "Você pode selecionar múltiplos valores para cada classe e visualizar os resultados em tempo real.")

    # multiselect → permite escolher várias classes
    # options → classes existentes no dataset (já traduzidas)
    # default → todas selecionadas ao abrir a página
    classes_selecionadas = st.multiselect(
        'Selecione as classes dos veículos:',
        options = df_traduzido['Classe'].unique(),
        default = df_traduzido['Classe'].unique()
    )

# ─── APLICAÇÃO DO FILTRO ─────────────────────────────────────────────────────

# isin(...) → mantém apenas as linhas cuja classe está entre as selecionadas
df_filtrado = df_traduzido[df_traduzido['Classe'].isin(classes_selecionadas)]

# Totais usados nos indicadores e no texto de contagem
total_registros = len(df_traduzido)     # Todos os registros do dataset
registros_exibidos = len(df_filtrado)   # Registros que passaram pelo filtro

# ─── INDICADORES ─────────────────────────────────────────────────────────────

m1, m2 = st.columns(2)

with m1:
    with st.container(border=True):
        # Terceiro parâmetro de st.metric → "delta", exibido abaixo do valor        
        st.metric(
            '📄 Total de Registros',
            f'{total_registros:,}'.replace(',', '.'),   # Padrão brasileiro: 1.000
            f'{total_registros / total_registros:.0%}'
        ) # Total de linhas

with m2:
    with st.container(border=True):
        # Percentual do dataset que está selecionado pelo filtro        
        st.metric(
            '📊 Registros Selecionados',
            f'{registros_exibidos:,}'.replace(',', '.'),
            f'{registros_exibidos / total_registros:.0%}', delta_color="off"
        ) # Total de linhas selecionadas

# ─── TABELA DE DADOS FILTRADOS ───────────────────────────────────────────────

with st.container(border=True):

    st.subheader("📋 Amostra e Detalhamento dos Dados")
    st.markdown(f"Exibindo **{registros_exibidos}** de **{total_registros}** registros disponíveis.")

    # Tabela interativa: permite ordenar, pesquisar e redimensionar colunas
    # hide_index=True → esconde o índice numérico | height=450 → altura em pixels
    st.dataframe(
        df_filtrado,
        width='stretch',
        hide_index=True,
        height=450
    )

    # Duas colunas [6, 1]: a da esquerda fica vazia e empurra o botão para a direita
    col_esquerda, col_direita = st.columns([6, 1])

    with col_direita:
        # Converte o DataFrame filtrado em CSV (em memória) e codifica em bytes
        # index=False → não inclui a coluna de índice no arquivo
        csv = df_filtrado.to_csv(index=False).encode('utf-8')

        # Botão que baixa o CSV com os dados filtrados
        st.download_button(
            label="📄 Baixar como CSV",
            data=csv,
            file_name='dados_filtrados.csv',
            mime='text/csv',
            width='stretch'
        )

# ─── ESTATÍSTICAS E FREQUÊNCIAS ──────────────────────────────────────────────

with st.container(border=True):
    st.subheader("📊 Estatísticas e Frequências por Categoria")

    # Duas colunas [1, 2]: tabela de frequência (menor) e resumo estatístico (maior)
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown('Frequencia absoluta por Classes')

        # value_counts() → conta quantos registros existem em cada classe
        # reset_index() → transforma o resultado em DataFrame
        df_frequencia = df_filtrado['Classe'].value_counts().reset_index()

        # Renomeia as colunas geradas pelo pandas        
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

        # describe(include='all') → resumo de todas as colunas, inclusive as de texto
        # Para colunas de texto devolve: count, unique, top e freq
        # .T → transpõe: cada atributo vira uma linha
        # rename → traduz os nomes das estatísticas para português
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