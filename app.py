# ─── IMPORTAÇÕES ──────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_data,  model
from charts.charts import*

# ─── CONFIGURAÇÃO DA PÁGINA ───────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title='Classificação de Veículos',
    page_icon="🚗",
    layout='wide'
)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

with st.sidebar:
    # Título principal da barra lateral
    st.markdown("### 🚗 Painel de Navegação")
    st.markdown("---")

    # Informações do Dataset com visual refinado
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

    # Objetivo da Aplicação estruturado
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

    # Espaçador vertical dinâmico para empurrar o rodapé para o fim da barra
    st.markdown("<br>" *6, unsafe_allow_html=True) 
    
    st.divider() # Linha divisória horizontal limpa
    
    # Rodapé profissionalizado e alinhado com o tema escuro
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

# ─── LEITURA DOS DADOS ────────────────────────────────────────────────────────

df = load_data()
encoder, modelo, y_pred, y_test = model(df)

# ─── DICIONÁRIOS DE TRADUÇÃO ─────────────────────────────────────────────────
# Criados APÓS o treinamento para não interferir na ordem ordinal aprendida
# Servem APENAS para exibição ao usuário nos selectboxes

traducao = {
    # Tradução dos valores de preço de compra
    'buying': {
        'vhigh': 'Muito Alto', 'high': 'Alto',
        'med':   'Médio',      'low':  'Baixo'
    },
    # Tradução dos valores de custo de manutenção
    'maint': {
        'vhigh': 'Muito Alta', 'high': 'Alta',
        'med':   'Média',      'low':  'Baixa'
    },
    # Tradução dos valores de número de portas
    'doors': {
        '2': '2 Portas',  '3': '3 Portas',
        '4': '4 Portas',  '5more': '5 ou mais'
    },
    # Tradução dos valores de capacidade de passageiros
    'persons': {
        '2': '2 Pessoas', '4': '4 Pessoas',
        'more': 'Mais de 4'
    },
    # Tradução dos valores de tamanho do porta-malas
    'lug_boot': {
        'small': 'Pequeno', 'med': 'Médio',
        'big':   'Grande'
    },
    # Tradução dos valores de nível de segurança
    'safety': {
        'low': 'Baixa', 'med': 'Média',
        'high': 'Alta'
    },
    'class':{
        'unacc': 'Inaceitável',
        'acc': 'Aceitável',
        'good': 'Bom',
        'vgood': 'Muito Bom'

    }
}

# Tradução reversa — converte o valor exibido em português de volta para o original em inglês
# Necessário para passar o valor correto ao encoder.transform() antes da previsão
# {v: k for k, v in mapa.items()} → inverte chave e valor de cada dicionário
traducao_reversa = {
    col: {v: k for k, v in mapa.items()}
    for col, mapa in traducao.items()
}

# Tradução do resultado da previsão — converte a classe prevista para português
traducao_resultado = {
    'unacc': 'Inaceitável',
    'acc':   'Aceitável',
    'good':  'Bom',
    'vgood': 'Muito Bom'
}

df_traduzido = df.copy()

for key, value in traducao.items():
    if key in df_traduzido.columns:
        df_traduzido[key] = df_traduzido[key].astype(str).map(value).fillna(df_traduzido[key])

# Dicionário para traduzir o nome das colunas
traducao_colunas = {
    'buying': 'Preço de Compra',
    'maint': 'Manutenção',
    'doors': 'Portas',
    'persons': 'Capacidade',
    'lug_boot': 'Porta-Malas',
    'safety': 'Segurança',
    'class': 'Classe'
}

df_traduzido = df_traduzido.rename(columns=traducao_colunas)

# ─── TÍTULO ────────────────────────────────────────────────────────────────

st.title('🚗 Classificação de Veículos com Naive Bayes')

st.caption('Utilizando o Conjunto de dados de avaliação de automóveis')

acuracia, precisao, recall, f1 = metricas(y_test, y_pred)

# ─── KPIS DADOS ────────────────────────────────────────────────────────────

col_metrics, col_grafico = st.columns([2, 1])

with col_metrics:
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric('📄 Registros', len(df))
    with m2:
        st.metric('📊 Atributos', len(df.columns.drop('class')))
    with m3:
        st.metric('🏷️ Classes', df['class'].nunique())
    with m4:
        st.metric('Ausentes', int(df.isna().sum().sum()))

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
    fig_gauge(acuracia)
    

# ─── LINHA PRINCIAPAL ────────────────────────────────────────────

col1, col2, col3 = st.columns([1.6, 1, 1])

with col1:
    st.subheader('📄 Amostra de Dados')
    st.dataframe(
        df_traduzido.head(10),
        use_container_width=True,
        hide_index=True,
        height=400
    )

with col2:
    st.subheader('📊 Distribuição das Classes')
    bar_chart(df_traduzido)    

with col3:
    st.subheader('📈 Matrix de Confusão')
    matriz_confusao(y_test, y_pred)

col_rel1, col_rel2 = st.columns([1, 2])

with col_rel1:
    st.subheader('📝 Relatório de Classificação')
    
    report_df = metrics_report(y_test, y_pred)

    st.dataframe(
        report_df.style.format({
            'Precisão': '{:.2f}%',
            'Recall': '{:.2f}%',
            'F1-Score': '{:.2f}%',
            'Suporte': '{:.0f}'
        })
    )
    

with col_rel2:
    st.subheader("🚗 Nova Previsão")

    with st.form('previsão'):
        r1, r2, r3 = st.columns(3)
        
        with r1:
            preco = st.selectbox('Preço', [traducao['buying'][v] for v in df['buying'].unique()])
            manutencao = st.selectbox('Manutenção', [traducao['maint'][v] for v in df['maint'].unique()])

        with r2:
            portas = st.selectbox('Portas', [traducao['doors'][v] for v in df['doors'].unique()])
            pessoas = st.selectbox('Capacidade de Passageiros', [traducao['persons'][v] for v in df['persons'].unique()])

        with r3:
            porta_malas = st.selectbox('Porta Malas', [traducao['lug_boot'][v] for v in df['lug_boot'].unique()])
            segurança = st.selectbox('Segurança', [traducao['safety'][v]   for v in df['safety'].unique()])            

        processar = st.form_submit_button('🔮 Fazer Previsão')

        
        # ─── RESULTADO ────────────────────────────────────────────
        input_features_traduzidos = [preco, manutencao, portas, pessoas, porta_malas, segurança]

        if processar:   
        # Lista com os nomes originais das colunas do dataset
            colunas = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
                
        # Converte os valores traduzidos de volta para os originais em inglês
        # zip(colunas, input_features_traduzidos) → emparelha coluna com valor selecionado
        # traducao_reversa[col][val] → busca o valor original pelo valor traduzido
        # IMPORTANTE: esse passo é obrigatório pois o encoder foi treinado com valores em inglês
            input_original = [
                traducao_reversa[col][val]
                for col, val in zip(colunas, input_features_traduzidos)
            ]
                
            # Converte a lista de valores originais em DataFrame
            # columns=df.columns.drop('class') → usa os mesmos nomes de colunas do dataset
            input_df = pd.DataFrame([input_original], columns=df.columns.drop('class'))
                
            # Aplica o mesmo encoder usado no treino para transformar as categorias em números
            # transform (não fit_transform) → usa o mapeamento já aprendido no treino
            input_encoder = encoder.transform(input_df)
                
            # Faz a previsão — retorna o código numérico da classe prevista
            predict_encoded = modelo.predict(input_encoder)
                
            # Converte o código numérico de volta para o nome da classe original em inglês
            # cat.categories → lista de categorias ordenadas ['acc', 'good', 'unacc', 'vgood']
            # [predict_encoded][0] → acessa a categoria pelo índice e pega o primeiro resultado
            previsao_original = df['class'].astype('category').cat.categories[predict_encoded][0]
                
            # Traduz o resultado para português
            # .get(previsao_original, previsao_original) → usa o original se não encontrar tradução
            previsao_traduzida = traducao_resultado.get(previsao_original, previsao_original)

            # Exibe o resultado traduzido em um card verde de sucesso
            if previsao_traduzida == 'Inaceitável':
                st.error(f'Resultado da previsão: 🔴 {previsao_traduzida}')
            elif previsao_traduzida == 'Aceitável':
                st.warning(f'Resultado da previsão: 🟡 {previsao_traduzida}')            
            elif previsao_traduzida == 'Bom':
                st.success(f'Resultado da previsão: 🟢 {previsao_traduzida}')            
            else :
                st.info(f'Resultado da previsão: 💎 {previsao_traduzida}')