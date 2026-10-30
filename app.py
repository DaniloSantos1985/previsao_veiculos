# ─── IMPORTAÇÕES ─────────────────────────────────────────────────────────────

import streamlit as st                  # Framework para criação da interface web
import pandas as pd                     # Manipulação de dados

# load_data → lê o CSV | model → treina o Naive Bayes e gera as previsões de teste
from utils.data_loader import load_data, model

# Importa tudo de charts.py: metricas, metrics_report, fig_gauge, bar_chart, matriz_confusao
# Obs.: import * funciona, mas importar as funções pelo nome deixa o código mais claro
from charts.charts import *

# ─── CONFIGURAÇÃO DA PÁGINA ──────────────────────────────────────────────────

# Título da aba, ícone e layout em tela cheia
st.set_page_config(
    page_title='Classificação de Veículos',
    page_icon="🚗",
    layout='wide'
)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

# Tudo dentro do with st.sidebar é desenhado na barra lateral
with st.sidebar:
    st.markdown("### 🚗 Painel de Navegação")
    st.markdown("---")

    # Card HTML com informação do dataset
    # border-left → faixa colorida na lateral esquerda do card
    # unsafe_allow_html=True → necessário para o Streamlit renderizar HTML
    st.markdown(
        """
        <div style='background-color: #1e2129; padding: 12px; border-radius: 8px; border-left: 4px solid #e694ff;'>
            <p style='margin: 0; font-size: 13px; color: #a0a0a0;'><b>DATASET ATIVO</b></p>
            <p style='margin: 4px 0 0 0; font-size: 14px; color: #ffffff;'>Conjunto de dados de avaliação de automóveis</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Explica o objetivo do modelo e as 4 classes possíveis
    st.markdown("### 🎯 Objetivo do Modelo")
    st.markdown(
        """
        Classificar o nível de aceitação dos veículos com base em critérios técnicos:
        - 🔴 **Inaceitável** (`unacc`)
        - 🟡 **Aceitável** (`acc`)
        - 🟢 **Bom** (`good`)
        - 💎 **Muito Bom** (`vgood`)
        """
    )

    # Espaçador vertical: repete <br> 6 vezes para empurrar o rodapé para baixo
    st.markdown("<br>" * 6, unsafe_allow_html=True)

    st.divider()  # Linha divisória horizontal

    # Rodapé centralizado com HTML e CSS inline
    st.markdown(
        """
        <div style='text-align: center; color: #888888; font-size: 12px; padding-bottom: 14px;'>
            <p style='margin: 2px 0;'><b>Dashboard de Qualidade Veicular</b></p>
            <p style='margin: 2px 0;'>Desenvolvido com Python & Streamlit</p>
            <p style='margin: 2px 0; color: #666666;'>© 2026 - Todos os direitos reservados</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ─── LEITURA DOS DADOS E TREINO DO MODELO ────────────────────────────────────

# Lê o CSV original (colunas e valores em inglês)
df = load_data()

# Desempacota as variáveis do model
encoder, modelo, y_test, y_pred = model(df)

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

# Tradução do resultado da previsão
# Repete o conteúdo de traducao['class']; poderia usar traducao['class'] direto
traducao_resultado = {
    'unacc': 'Inaceitável',
    'acc':   'Aceitável',
    'good':  'Bom',
    'vgood': 'Muito Bom'
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

# ─── TÍTULO ──────────────────────────────────────────────────────────────────

st.title('🚗 Classificação de Veículos com Naive Bayes')
st.caption('Utilizando o Conjunto de dados de avaliação de automóveis')

# Calcula as 4 métricas gerais do modelo (função de charts.py)
acuracia, precisao, recall, f1 = metricas(y_test, y_pred)

# ─── KPIs ────────────────────────────────────────────────────────────────────

# Duas colunas: cards de métricas (2/3 da largura) e medidor (1/3)
col_metrics, col_grafico = st.columns([2, 1])

with col_metrics:
    # Primeira linha: informações do dataset
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric('📄 Registros', len(df))                              # Total de linhas
    with m2:
        st.metric('📊 Atributos', len(df.columns.drop('class')))        # Colunas, sem a classe
    with m3:
        st.metric('🏷️ Classes', df['class'].nunique())                  # Classes distintas
    with m4:
        st.metric('Ausentes', int(df.isna().sum().sum()))               # Total de valores nulos

    # Segunda linha: métricas do modelo, formatadas como porcentagem (:.2%)
    m5, m6, m7, m8 = st.columns(4)
    with m5:
        st.metric('🎯 Acurácia', f'{acuracia:.2%}')
    with m6:
        st.metric('✅ Precisão', f'{precisao:.2%}')
    with m7:
        st.metric('Recall', f'{recall:.2%}')
    with m8:
        st.metric('F1-Score', f"{f1:.2%}")

with col_grafico:
    # Medidor circular da acurácia (função de charts.py)
    fig_gauge(acuracia)

# ─── LINHA PRINCIPAL: TABELA E GRÁFICOS ──────────────────────────────────────

# Três colunas: a tabela é mais larga (1.6) que os dois gráficos (1 cada)
col1, col2, col3 = st.columns([1.6, 1, 1])

with col1:
    st.subheader('📄 Amostra de Dados')
    # Primeiras 10 linhas já traduzidas, sem a coluna de índice    
    st.dataframe(
        df_traduzido.head(10),
        width='stretch',
        hide_index=True,
        height=380
    )

with col2:
    st.subheader('📊 Distribuição das Classes')
    bar_chart(df_traduzido)   # Usa a coluna 'Classe' traduzida

with col3:
    st.subheader('📈 Matrix de Confusão')
    matriz_confusao(y_test, y_pred)

# ─── LINHA INFERIOR: RELATÓRIO E PREVISÃO ────────────────────────────────────

col_rel1, col_rel2 = st.columns([1, 2])

with col_rel1:
    st.subheader('📝 Relatório de Classificação')

    # Relatório por classe (precisão, recall, F1 e suporte), já traduzido
    report_df = metrics_report(y_test, y_pred)
    
    st.dataframe(
        report_df.style.format({
            'Precisão': '{:.2%}',
            'Recall':   '{:.2%}',
            'F1-Score': '{:.2%}',
            'Suporte':  '{:.0f}'     # Suporte é contagem, sem casas decimais
        })
    )

with col_rel2:
    st.subheader("🚗 Nova Previsão")

    # st.form agrupa os campos: a página só é reprocessada ao clicar no botão
    with st.form('previsão'):
        r1, r2, r3 = st.columns(3)

        ordem_buying_maint = ['low', 'med', 'high', 'vhigh'] 
        ordem_doors = ['2', '3', '4', '5more']       
        ordem_people = ['2', '4', 'more']
        ordem_lug_boot = ['small', 'med', 'big']
        ordem_safety = ['low', 'med', 'high']

        # Cada selectbox mostra as opções traduzidas
        # df[col].unique() → valores únicos, na ordem em que aparecem no CSV
        # A ordem fica embaralhada (ex.: Alto, Muito Alto, Baixo...); definir uma
        # lista fixa (Baixo, Médio, Alto, Muito Alto) melhora a experiência
        with r1:            
            preco      = st.selectbox('Preço',      [traducao['buying'][v] for v in ordem_buying_maint])            
            manutencao = st.selectbox('Manutenção', [traducao['maint'][v]  for v in ordem_buying_maint])

        with r2:            
            pessoas = st.selectbox('Capacidade de Passageiros', [traducao['persons'][v] for v in ordem_people])
            portas  = st.selectbox('Portas',                    [traducao['doors'][v]   for v in ordem_doors])

        with r3:            
            porta_malas = st.selectbox('Porta Malas', [traducao['lug_boot'][v] for v in ordem_lug_boot])
            segurança   = st.selectbox('Segurança',   [traducao['safety'][v]   for v in ordem_safety])

        # Botão de envio do formulário: retorna True quando clicado
        processar = st.form_submit_button('🔮 Fazer Previsão')

        # ─── RESULTADO ───────────────────────────────────────────────────────

        # Valores escolhidos, ainda em português, na mesma ordem das colunas do CSV
        input_features_traduzidos = [preco, manutencao, portas, pessoas, porta_malas, segurança]

        if processar:
            # Nomes originais das colunas (em inglês), na mesma ordem da lista acima
            colunas = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

            # Tradução reversa: português → inglês
            # zip(colunas, input_features_traduzidos) → emparelha coluna e valor escolhido
            # traducao_reversa[col][val] → busca o valor original em inglês
            # Obrigatório porque o encoder só conhece os valores em inglês
            input_original = [
                traducao_reversa[col][val]
                for col, val in zip(colunas, input_features_traduzidos)
            ]

            # Monta um DataFrame de 1 linha com os mesmos nomes de coluna do treino
            input_df = pd.DataFrame([input_original], columns=df.columns.drop('class'))

            # transform (não fit_transform) → aplica o mapeamento aprendido no treino
            input_encoder = encoder.transform(input_df)

            # Previsão: retorna o código numérico da classe (0, 1, 2 ou 3)
            predict_encoded = modelo.predict(input_encoder)

            # Converte o código de volta para o nome da classe em inglês
            # cat.categories → ['acc', 'good', 'unacc', 'vgood'] (ordem alfabética)
            previsao_original = df['class'].astype('category').cat.categories[predict_encoded][0]

            # Traduz para português; .get usa o valor original se não achar tradução
            previsao_traduzida = traducao_resultado.get(previsao_original, previsao_original)

            # Cada classe aparece em um card de cor diferente
            if previsao_traduzida == 'Inaceitável':
                st.error(f'Resultado da previsão: 🔴 {previsao_traduzida}')      # Vermelho
            elif previsao_traduzida == 'Aceitável':
                st.warning(f'Resultado da previsão: 🟡 {previsao_traduzida}')    # Amarelo
            elif previsao_traduzida == 'Bom':
                st.success(f'Resultado da previsão: 🟢 {previsao_traduzida}')    # Verde
            else:
                st.info(f'Resultado da previsão: 💎 {previsao_traduzida}')       # Azul (Muito Bom)