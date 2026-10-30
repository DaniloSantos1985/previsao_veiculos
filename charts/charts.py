# ─── IMPORTAÇÕES ─────────────────────────────────────────────────────────────

import pandas as pd                                  # Manipulação e leitura de dados
import streamlit as st                               # Framework para criação da interface web
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def metricas(y_test, y_pred):
    # Calcula a acurácia — percentual de previsões corretas
    acuracia = accuracy_score(y_test, y_pred)
    
    precisao = precision_score(y_test, y_pred, average='macro', zero_division=0)
    
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)    

    return acuracia, precisao, recall, f1


def metrics_report(y_test, y_pred):
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()

    traducao_colunas = {
        'precision': 'Precisão',
        'recall': 'Recall',
        'f1-score': 'F1-Score',
        'support': 'Suporte'
    }

    traducao_linhas = {
        '0': 'Aceitável',
        '1': 'Bom',
        '2': 'Inaceitável',
        '3': 'Muito Bom',
        'accuracy': 'Acurácia',
        'macro avg': 'Média Macro',
        'weighted avg': 'Média Ponderada'
    }

    report_df = report_df.rename(columns=traducao_colunas)
    report_df = report_df.rename(index=traducao_linhas)

    return report_df


def fig_gauge(acuracia):
    """
    Gera o gráfico de medidor (gauge) com o título visível e alinhado às métricas.
    """
    fig = go.Figure(
        go.Indicator(
            mode='gauge+number',
            value=acuracia * 100,  # Converte para porcentagem
            number={
                'suffix': "%",
                'font': {'size': 28}  # Tamanho proporcional para caber na altura
            },
            # Configuração explícita do título para garantir que apareça
            title={
                'text': "Acurácia Geral do Modelo",
                'font': {'size': 14, 'color': 'white'}
            },
            gauge={
                'axis': {
                    'range': [0, 100], 
                    'tickwidth': 1, 
                    'tickcolor': "white",
                    'tickfont': {'size': 9}
                },
                'bar': {'color': '#2e7d32'}, 
                'bgcolor': "#262730",
                'borderwidth': 1,
                'bordercolor': "#374151",
                'steps': [
                    {"range": [0, 60], "color": "#5c2b29"},   
                    {"range": [60, 80], "color": "#5e5b22"},  
                    {"range": [80, 100], "color": "#1b4d3e"}  
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 3},
                    'thickness': 0.75,
                    'value': acuracia * 100
                }
            }
        )
    )

    fig.update_layout(
        margin=dict(l=15, r=15, t=35, b=5),
        height=160, # Altura ajustada para alinhar visualmente com as duas linhas de métricas                
    )
    
    # Ajustes visuais para o gráfico encaixar perfeitamente ao lado das métricas
    st.plotly_chart(
        fig,
        width='stretch'
    )


def bar_chart(df):
    dist = (
        df['Classe']
        .value_counts()
        .reset_index()
    )
    
    dist.columns = ('Classe', 'Quantidade')
    
    fig_dist = px.bar(
        dist,
        x='Classe',
        y='Quantidade',
        color='Quantidade',          # Usa a quantidade numérica para gerar o degradê
        color_continuous_scale='Greens', # Aplica a mesma escala de verde da matriz
        text='Quantidade'        
    )

    # Ajustes visuais de layout para o dashboard e tema escuro
    fig_dist.update_layout(
        title={             
            'y': 0.93,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        height=380,
        margin=dict(t=10, l=20, r=20, b=30),                
        showlegend=False,            # Remove a legenda lateral
        coloraxis_showscale=False    # Oculta a barra de escala contínua lateral para manter limpo
    )
    
    # Remove o título do eixo X (horizontal)
    fig_dist.update_xaxes(title_text="")

    st.plotly_chart(
        fig_dist,
        width='stretch'
    )


def matriz_confusao(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    classes_veiculos = [
        'Aceitável',   
        'Bom',         
        'Inaceitável', 
        'Muito Bom'    
    ]    

    fig_cm = px.imshow(
        cm,
        text_auto=True,
        aspect='auto',
        color_continuous_scale='Greens',
        labels=dict(x='Classe Predita', y='Classe Real', color='Contagem'),
        x=classes_veiculos,
        y=classes_veiculos
    )

    # Ajustes visuais de layout para proporcionalidade perfeita
    fig_cm.update_layout(        
        height=380,
        margin=dict(t=10, l=20, r=20, b=30),                
        coloraxis=dict(
            cmin=0,
            cmax=80,
            colorbar=dict(
                title='',                
                thickness=10,      # Espessura equilibrada da barra
                len=1.0,          # Altura proporcional à grade da matriz                
            )
        )
    )

    st.plotly_chart(fig_cm, use_container_width=True)