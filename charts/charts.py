# ─── IMPORTAÇÕES ─────────────────────────────────────────────────────────────

import pandas as pd                  # Manipulação e leitura de dados
import streamlit as st               # Framework para criação da interface web
import plotly.express as px          # Gráficos de alto nível — barras e matriz de confusão
import plotly.graph_objects as go    # Gráficos de baixo nível — indicador (gauge)

# Funções do scikit-learn para avaliar o desempenho do modelo
from sklearn.metrics import (
    accuracy_score,           # Percentual de previsões corretas
    precision_score,          # Dos que o modelo previu como X, quantos eram X
    recall_score,             # Dos que realmente eram X, quantos o modelo encontrou
    f1_score,                 # Média harmônica entre precisão e recall
    confusion_matrix,         # Tabela de acertos e erros por classe
    classification_report     # Relatório completo por classe
)


# ─── FUNÇÃO: MÉTRICAS GERAIS ─────────────────────────────────────────────────

def metricas(y_test, y_pred):
    """
    Calcula as métricas gerais de avaliação do modelo.

    Parâmetros:
    y_test: valores reais da classe (conjunto de teste)
    y_pred: valores previstos pelo modelo

    Retorna:
    tuple: acurácia, precisão, recall e F1-Score
    """

    # Acurácia — percentual de previsões corretas sobre o total
    acuracia = accuracy_score(y_test, y_pred)

    # average='macro' → calcula a métrica de cada classe e tira a média simples
    # Trata todas as classes com o mesmo peso, mesmo as menos frequentes
    # zero_division=0 → evita erro quando uma classe não recebe nenhuma previsão
    precisao = precision_score(y_test, y_pred, average='macro', zero_division=0)

    # Recall com média macro
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)

    # F1-Score com média macro
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    return acuracia, precisao, recall, f1


# ─── FUNÇÃO: RELATÓRIO DE CLASSIFICAÇÃO ──────────────────────────────────────

def metrics_report(y_test, y_pred):
    """
    Gera o relatório de classificação por classe, traduzido para português.

    Retorna:
    pd.DataFrame: tabela com precisão, recall, F1-Score e suporte por classe
    """

    # output_dict=True → retorna o relatório como dicionário em vez de texto
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Converte o dicionário em DataFrame e transpõe
    # transpose() → classes ficam nas linhas e métricas nas colunas
    report_df = pd.DataFrame(report).transpose()

    # Dicionário de tradução dos nomes das colunas
    traducao_colunas = {
        'precision': 'Precisão',
        'recall':    'Recall',
        'f1-score':  'F1-Score',
        'support':   'Suporte'      # Quantidade de registros reais de cada classe
    }

    # Dicionário de tradução dos nomes das linhas
    # '0' a '3' são os códigos numéricos das classes gerados por cat.codes
    # (ordem alfabética original: acc, good, unacc, vgood)
    traducao_linhas = {
        '0':            'Aceitável',
        '1':            'Bom',
        '2':            'Inaceitável',
        '3':            'Muito Bom',
        'accuracy':     'Acurácia',
        'macro avg':    'Média Macro',
        'weighted avg': 'Média Ponderada'
    }

    # Aplica as traduções nas colunas e nas linhas
    report_df = report_df.rename(columns=traducao_colunas)
    report_df = report_df.rename(index=traducao_linhas)

    return report_df


# ─── FUNÇÃO: GRÁFICO DE MEDIDOR (GAUGE) ──────────────────────────────────────

def fig_gauge(acuracia):
    """
    Gera o gráfico de medidor (gauge) com a acurácia do modelo.
    Ele fica ao lado dos cards de métricas, por isso tem altura reduzida.
    """

    fig = go.Figure(
        go.Indicator(
            mode='gauge+number',           # Exibe o ponteiro e o número central
            value=acuracia * 100,          # Converte de 0–1 para percentual 0–100

            # Número central do medidor
            number={
                'suffix': "%",             # Adiciona o símbolo de % após o valor
                'font': {'size': 28}       # Tamanho proporcional à altura do gráfico
            },

            # Título exibido acima do medidor
            title={
                'text': "Acurácia Geral do Modelo",
                'font': {'size': 14, 'color': 'white'}
            },

            gauge={
                # Configuração do eixo do medidor
                'axis': {
                    'range': [0, 100],     # Escala de 0 a 100%
                    'tickwidth': 1,
                    'tickcolor': "white",
                    'tickfont': {'size': 9}
                },
                'bar': {'color': '#2e7d32'},      # Cor da barra de progresso (verde)
                'bgcolor': "#262730",             # Cor de fundo do medidor (tema escuro)
                'borderwidth': 1,
                'bordercolor': "#374151",

                # Faixas coloridas de desempenho
                'steps': [
                    {"range": [0, 60],   "color": "#5c2b29"},  # Vermelho — ruim
                    {"range": [60, 80],  "color": "#5e5b22"},  # Amarelo — regular
                    {"range": [80, 100], "color": "#1b4d3e"}   # Verde — bom
                ],

                # Linha branca que marca o valor atual
                'threshold': {
                    'line': {'color': "white", 'width': 3},
                    'thickness': 0.75,
                    'value': acuracia * 100
                }
            }
        )
    )

    # Margens reduzidas e altura fixa para alinhar com as duas linhas de métricas
    fig.update_layout(
        margin=dict(l=15, r=15, t=35, b=5),
        height=160,
    )

    # width='stretch' → ocupa toda a largura da coluna
    st.plotly_chart(fig, width='stretch')


# ─── FUNÇÃO: GRÁFICO DE BARRAS — DISTRIBUIÇÃO DAS CLASSES ────────────────────

def bar_chart(df):
    """
    Gera o gráfico de barras com a quantidade de registros por classe.
    Espera um DataFrame com a coluna 'Classe' já traduzida.
    """

    # value_counts() → conta quantos registros existem em cada classe
    # reset_index() → transforma o resultado em DataFrame
    dist = df['Classe'].value_counts().reset_index()

    # Renomeia as colunas para nomes claros
    dist.columns = ('Classe', 'Quantidade')

    fig_dist = px.bar(
        dist,
        x='Classe',
        y='Quantidade',
        color='Quantidade',               # Cor proporcional à quantidade (degradê)
        color_continuous_scale='Greens',  # Mesma escala de verde da matriz de confusão
        text='Quantidade'                 # Exibe o valor sobre cada barra
    )

    fig_dist.update_layout(
        # Configuração de título sem o campo 'text' — não exibe nada
        # O título já aparece no st.subheader da página
        title={
            'y': 0.93,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        height=380,                          # Mesma altura do dataframe e da matriz
        margin=dict(t=10, l=20, r=20, b=30), # Margem superior mínima
        showlegend=False,                    # Remove a legenda lateral
        coloraxis_showscale=False            # Oculta a barra de escala de cores
    )

    # Remove o título do eixo X para deixar o gráfico mais limpo
    fig_dist.update_xaxes(title_text="")

    st.plotly_chart(fig_dist, width='stretch')


# ─── FUNÇÃO: MATRIZ DE CONFUSÃO ──────────────────────────────────────────────

def matriz_confusao(y_test, y_pred):
    """
    Gera a matriz de confusão — linhas são as classes reais
    e colunas são as classes previstas pelo modelo.
    """

    # Calcula a matriz: cada célula mostra quantos registros de uma
    # classe real foram previstos como outra classe
    cm = confusion_matrix(y_test, y_pred)

    # Nomes das classes na ordem dos códigos numéricos (0, 1, 2, 3)
    classes_veiculos = [
        'Aceitável',
        'Bom',
        'Inaceitável',
        'Muito Bom'
    ]

    fig_cm = px.imshow(
        cm,
        text_auto=True,                    # Exibe o valor dentro de cada célula
        aspect='auto',                     # Células preenchem toda a área disponível
        color_continuous_scale='Greens',   # Quanto maior o valor, mais escuro o verde
        labels=dict(x='Classe Predita', y='Classe Real', color='Contagem'),
        x=classes_veiculos,                # Rótulos do eixo X (classe predita)
        y=classes_veiculos                 # Rótulos do eixo Y (classe real)
    )

    fig_cm.update_layout(
        height=380,                           # Mesma altura do dataframe e das barras
        margin=dict(t=10, l=20, r=20, b=30),
        coloraxis=dict(
            cmin=0,                           # Valor mínimo da escala de cores            
            colorbar=dict(
                title='',                     # Sem título na barra de cores
                thickness=10,                 # Espessura da barra lateral
                len=1.0,                      # Barra acompanha toda a altura da matriz
            )
        )
    )

    # ⚠️ Aqui está use_container_width, enquanto as outras funções usam width='stretch'
    st.plotly_chart(fig_cm, width='stretch')