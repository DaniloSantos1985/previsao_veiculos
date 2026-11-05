# ─── IMPORTAÇÕES ─────────────────────────────────────────────────────────────

import streamlit as st  # Framework para criação da interface web

# ─── CONFIGURAÇÃO DA PÁGINA ──────────────────────────────────────────────────

# set_page_config deve ser o primeiro comando Streamlit do arquivo
# page_icon aceita ícones do Material Symbols no formato ':material/nome:'
st.set_page_config(
    page_title='Classificação de Veículos',
    page_icon=':material/directions_car:',
    layout='wide',
)

# ─── CONTEÚDO DA PÁGINA (EDITE AQUI) ─────────────────────────────────────────

# Atributos usados como entrada do modelo: (nome exibido, valores possíveis)
ATRIBUTOS = [
    ('Preço de compra', 'Baixo, Médio, Alto ou Muito Alto'),
    ('Manutenção', 'custo Baixo, Médio, Alto ou Muito Alto'),
    ('Portas', '2, 3, 4 ou 5 ou mais'),
    ('Capacidade', '2, 4 ou mais de 4 pessoas'),
    ('Porta-malas', 'Pequeno, Médio ou Grande'),
    ('Segurança', 'Baixa, Média ou Alta'),
]

# Classes previstas pelo modelo: (marcador, nome exibido, código original)
CLASSES = [
    ('🔴', 'Inaceitável', 'unacc'),
    ('🟡', 'Aceitável', 'acc'),
    ('🟢', 'Bom', 'good'),
    ('💎', 'Muito Bom', 'vgood'),
]

# Páginas do app. Cada item tem:
# page      → caminho do arquivo, relativo a este arquivo (Principal.py)
# icone     → ícone do Material Symbols
# cor       → cor do ícone (hexadecimal)
# titulo    → texto exibido como link
# descricao → resumo de uma linha
# itens     → lista do que o usuário encontra na página
# ⚠️ Os nomes em 'page' precisam ser idênticos aos nomes dos arquivos em pages/
NAV_ITEMS = [
    {
        'page': 'pages/01_Visão_Geral.py',
        'icone': ':material/monitoring:',
        'cor': '#22c55e',
        'titulo': '01 — Visão Geral',
        'descricao': 'KPIs, distribuição das classes, indicadores e resumo do modelo.',
        'itens': [
            'Indicadores do dataset (registros, atributos, classes e valores ausentes)',
            'Métricas gerais do modelo e medidor de acurácia',
            'Distribuição das classes de veículos',
        ],
    },
    {
        'page': 'pages/02_Exploração_de_Dados.py',
        'icone': ':material/database:',
        'cor': '#3b82f6',
        'titulo': '02 — Exploração dos Dados',
        'descricao': 'EDA, estatísticas descritivas, filtros, correlações e qualidade dos dados.',
        'itens': [
            'Amostra e estatísticas descritivas do dataset',
            'Comportamento de cada atributo em relação às classes',
            'Filtros interativos e associações entre atributos',
            'Qualidade dos dados: nulos, duplicatas e rótulos conflitantes',
        ],
    },
    {
        'page': 'pages/03_Desempenho_do_Modelo.py',
        'icone': ':material/psychology:',
        'cor': '#a855f7',
        'titulo': '03 — Desempenho do Modelo',
        'descricao': 'Métricas, matriz de confusão, validação cruzada e comparação de algoritmos.',
        'itens': [
            'Acurácia, precisão, recall e F1-Score, geral e por classe',
            'Matriz de confusão para ver onde o modelo acerta e erra',
            'Validação cruzada para uma estimativa mais confiável',
            'Comparação do Naive Bayes com outros algoritmos',
        ],
    },
    {
        'page': 'pages/04_Predições.py',
        'icone': ':material/directions_car:',
        'cor': '#f59e0b',
        'titulo': '04 — Fazer Previsões',
        'descricao': 'Formulário, classificação, probabilidades e histórico de previsões.',
        'itens': [
            'Formulário para descrever um veículo e obter a classe prevista',
            'Probabilidade estimada para cada uma das quatro classes',
            'Histórico das previsões feitas na sessão',
        ],
    },    
    {
        'page': 'pages/05_Configurações.py',
        'icone': ':material/settings:',
        'cor': '#9ca3af',
        'titulo': '05 — Configurações',
        'descricao': 'Versão do modelo, parâmetros, informações do treinamento e configuração da aplicação.',
        'itens': [
            'Versão do modelo e parâmetros utilizados',
            'Informações do treinamento (divisão treino/teste, semente)',
            'Configurações da aplicação',
        ],
    },
]

# ─── ESTILO ──────────────────────────────────────────────────────────────────

# Cada container recebe uma 'key' (nav_0, nav_1...) e o Streamlit adiciona
# a classe CSS 'st-key-nav_0', 'st-key-nav_1'... ao elemento.
# Gera uma regra por item para pintar o ícone com a cor definida em NAV_ITEMS
# e aumenta a fonte do título do link
css = '\n'.join(
    f'.st-key-nav_{i} [data-testid="stIconMaterial"] {{ color: {item["cor"]}; }}'
    for i, item in enumerate(NAV_ITEMS)
)

st.markdown(
    f"""
    <style>
    {css}
    [class*="st-key-nav_"] a p {{ font-size: 1.15rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── CABEÇALHO ───────────────────────────────────────────────────────────────

st.header(':material/dashboard: Dashboard de Classificação de Veículos')
st.caption('Navegação principal da aplicação')

# ─── SOBRE O PROJETO ─────────────────────────────────────────────────────────

st.subheader('Sobre o projeto')

st.markdown(
    """
    Este dashboard apresenta uma solução de **classificação de veículos**: a partir de
    seis características técnicas, um modelo de aprendizado de máquina
    (**Naive Bayes Categórico**) estima o nível de aceitação de um automóvel em quatro classes.

    O projeto percorre o ciclo de uma análise de dados: exploração e qualidade dos dados,
    treinamento e avaliação do modelo, previsões com probabilidades e geração de relatórios.
    """
)

# Resumo do dataset e do modelo
# Valores fixos por enquanto; depois podem ser lidos do dataset com load_data()
k1, k2, k3, k4 = st.columns(4)
k1.metric('Registros', '1.000')
k2.metric('Atributos', len(ATRIBUTOS))
k3.metric('Classes', len(CLASSES))
k4.metric('Algoritmo', 'Naive Bayes')

# Atributos de entrada e classes previstas, lado a lado
col_attr, col_cls = st.columns(2)

with col_attr:
    with st.container(border=True):
        st.markdown('**Atributos de entrada**')
        for nome, valores in ATRIBUTOS:
            st.markdown(f'- **{nome}:** {valores}')

with col_cls:
    with st.container(border=True):
        st.markdown('**Classes previstas**')
        for marcador, nome, codigo in CLASSES:
            st.markdown(f'- {marcador} **{nome}** (`{codigo}`)')

# ─── GUIA DAS PÁGINAS ────────────────────────────────────────────────────────

st.subheader('O que você encontra em cada página')
st.caption('Clique no título para abrir a página.')

for i, item in enumerate(NAV_ITEMS):
    # key única → permite aplicar o CSS de cor individual de cada ícone
    with st.container(key=f'nav_{i}'):
        # st.page_link cria um link clicável que navega para a página
        st.page_link(
            item['page'],
            label=f"**{item['titulo']}**",
            icon=item['icone'],
        )
        st.caption(item['descricao'])

        # Lista do que o usuário encontra na página
        st.markdown('\n'.join(f'- {texto}' for texto in item['itens']))

    # Linha divisória entre os itens (menos depois do último)
    if i < len(NAV_ITEMS) - 1:
        st.divider()

# ─── NOTA SOBRE OS DADOS ─────────────────────────────────────────────────────

st.divider()
st.caption(
    'Projeto de caráter educacional. O dataset é uma versão do Car Evaluation (UCI), com classes '
    'desbalanceadas, e os resultados não substituem uma avaliação real de veículos.'
)