# ═════════════════════════════════════════════════════════════════════════════
# Principal.py — Página inicial do Dashboard de Classificação de Veículos
#
# Função: apresentar o projeto e servir de menu para as demais páginas.
# Como executar (a partir da raiz do projeto): streamlit run Principal.py
# As páginas dentro da pasta pages/ são carregadas pelo Streamlit
# e ficam disponíveis no menu lateral e nos links desta página.
# ═════════════════════════════════════════════════════════════════════════════

# ─── IMPORTAÇÕES ─────────────────────────────────────────────────────────────

import streamlit as st  # Framework para criação da interface web

# ─── CONFIGURAÇÃO DA PÁGINA ──────────────────────────────────────────────────

# set_page_config deve ser o primeiro comando Streamlit do arquivo
# page_title → texto exibido na aba do navegador
# page_icon  → ícone da aba; aceita Material Symbols no formato ':material/nome:'
# layout     → 'wide' usa toda a largura da tela
st.set_page_config(
    page_title='Classificação de Veículos',
    page_icon=':material/directions_car:',
    layout='wide',
)

# ─── CONTEÚDO DA PÁGINA (EDITE AQUI) ─────────────────────────────────────────
# Os textos ficam em listas no topo do arquivo: para mudar o conteúdo
# basta editar estas listas, sem mexer no código de exibição mais abaixo

# Atributos usados como entrada do modelo: (nome exibido, valores possíveis)
ATRIBUTOS = [
    ('Preço de compra', 'Baixo, Médio, Alto ou Muito Alto'),
    ('Manutenção', 'custo Baixo, Médio, Alto ou Muito Alto'),
    ('Portas', '2, 3, 4 ou 5 ou mais'),
    ('Capacidade', '2, 4 ou mais de 4 pessoas'),
    ('Porta-malas', 'Pequeno, Médio ou Grande'),
    ('Segurança', 'Baixa, Média ou Alta'),
]

# Classes previstas pelo modelo: (marcador, nome exibido, código original no dataset)
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
# Esta lista alimenta duas coisas: o CSS dos ícones (seção ESTILO)
# e o guia de páginas (seção GUIA DAS PÁGINAS)
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

# Cada container do guia recebe uma 'key' (nav_0, nav_1...) e o Streamlit
# adiciona ao elemento a classe CSS 'st-key-nav_0', 'st-key-nav_1'...
# A expressão abaixo gera uma regra CSS por item, pintando o ícone
# (data-testid="stIconMaterial") com a cor definida em NAV_ITEMS
# As chaves duplicadas {{ }} são necessárias dentro de f-strings
# para escrever chaves literais do CSS
css = '\n'.join(
    f'.st-key-nav_{i} [data-testid="stIconMaterial"] {{ color: {item["cor"]}; }}'
    for i, item in enumerate(NAV_ITEMS)
)

# Injeta o CSS na página
# [class*="st-key-nav_"] → seleciona qualquer classe que contenha 'st-key-nav_'
# a p → texto do link; aumenta a fonte do título de cada página
# unsafe_allow_html=True → necessário para o Streamlit aceitar a tag <style>
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

# O título aceita ícones do Material Symbols no próprio texto
st.header(':material/dashboard: Dashboard de Classificação de Veículos')
st.caption('Navegação principal da aplicação')

# ─── SOBRE O PROJETO ─────────────────────────────────────────────────────────

st.subheader('Sobre o projeto')

# Texto explicativo em Markdown: ** ** deixa o trecho em negrito
st.markdown(
    """
    Este dashboard apresenta uma solução de **classificação de veículos**: a partir de
    seis características técnicas, um modelo de aprendizado de máquina
    (**Naive Bayes Categórico**) estima o nível de aceitação de um automóvel em quatro classes.

    O projeto percorre o ciclo de uma análise de dados: exploração e qualidade dos dados,
    treinamento e avaliação do modelo, previsões com probabilidades e geração de relatórios.
    """
)

# Resumo do dataset e do modelo em quatro indicadores
# Atributos e Classes são calculados a partir das listas acima
# Registros e Algoritmo são valores fixos por enquanto;
# depois podem ser lidos do dataset com load_data()
k1, k2, k3, k4 = st.columns(4)
k1.metric('Registros', '1.000')
k2.metric('Atributos', len(ATRIBUTOS))
k3.metric('Classes', len(CLASSES))
k4.metric('Algoritmo', 'Naive Bayes')

# Atributos de entrada e classes previstas, lado a lado
col_attr, col_cls = st.columns(2)

with col_attr:
    # border=True → desenha um cartão com borda em volta do conteúdo
    with st.container(border=True):
        st.markdown('**Atributos de entrada**')
        # Uma linha de lista (-) por atributo, com o nome em negrito
        for nome, valores in ATRIBUTOS:
            st.markdown(f'- **{nome}:** {valores}')

with col_cls:
    with st.container(border=True):
        st.markdown('**Classes previstas**')
        # Marcador colorido, nome da classe e código original entre crases
        for marcador, nome, codigo in CLASSES:
            st.markdown(f'- {marcador} **{nome}** (`{codigo}`)')

# ─── GUIA DAS PÁGINAS ────────────────────────────────────────────────────────

st.subheader('O que você encontra em cada página')
st.caption('Clique no título para abrir a página.')

# enumerate → fornece o índice i (usado na key do container e na divisória)
for i, item in enumerate(NAV_ITEMS):
    # A key única (nav_0, nav_1...) permite aplicar a cor individual de cada ícone
    with st.container(key=f'nav_{i}'):
        # st.page_link cria um link clicável que navega para a página
        # O label em ** ** fica em negrito
        st.page_link(
            item['page'],
            label=f"**{item['titulo']}**",
            icon=item['icone'],
        )
        # Descrição de uma linha, em texto pequeno, logo abaixo do link
        st.caption(item['descricao'])

        # Lista do que o usuário encontra na página
        # '\n'.join(...) monta uma lista Markdown com um item por linha
        st.markdown('\n'.join(f'- {texto}' for texto in item['itens']))

    # Linha divisória entre os itens (menos depois do último)
    if i < len(NAV_ITEMS) - 1:
        st.divider()

# ─── NOTA SOBRE OS DADOS ─────────────────────────────────────────────────────

st.divider()

# Aviso final: deixa claro o caráter educacional e a origem dos dados
st.caption(
    'Projeto de caráter educacional. O dataset é uma versão do Car Evaluation (UCI), com classes '
    'desbalanceadas, e os resultados não substituem uma avaliação real de veículos.'
)