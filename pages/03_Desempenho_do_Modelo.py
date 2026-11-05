import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from src.data_loader import load_data
from src.model_training import model
from src.charts import matriz_confusao, metrics_report

# Configuração da página
st.set_page_config(
    page_title="Desempenho do Modelo - Classificação de Veículos",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Desempenho e Avaliação do Modelo")
st.markdown("Análise estatística detalhada do classificador Naive Bayes, incluindo métricas por classe, matriz de confusão e validação.")

# Carrega os dados e executa o modelo treinado com balanceamento
df = load_data()
encoder, modelo, y_test, y_pred = model(df)

# Gera o relatório estruturado corretamente sem sobreposição da acurácia
report_df, acuracia_global = metrics_report(y_test, y_pred)

# SEÇÃO 1: MÉTRICAS GLOBAIS DE DESTAQUE
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label='Acurácia Global', 
        value=f'{acuracia_global:.2%}'.replace('.', ',')            
    )

with col2:        
    st.metric(
        label='Total de Registros de Testes', 
        value=f'{len(y_test)}'                
    )

with col3:        
    st.metric(
        label='Algoritmo Utilizado', 
        value='Categorial Naive Bayes'            
    )

with  st.container(border=True):
    col_esq,  col_dir = st.columns([1, 1])

    with col_esq:
        st.subheader("📊 Matriz de Confusão")
        st.markdown("Comparação entre as classes reais e as previsões do modelo no conjunto de teste.")
        # Chama a função modularizada da matriz de confusão
        matriz_confusao(y_test, y_pred)        

    with col_dir:
        st.subheader("📋 Relatório de Métricas por Classe")
        st.markdown("A tabela abaixo apresenta as métricas detalhadas de desempenho do modelo para cada classe, traduzidas para português.")
        # Gera o relatório estruturado corretamente sem sobreposição da acurácia
        st.dataframe(
            report_df.style.format({
                'Precisão': '{:.2%}',
                'Recall': '{:.2%}',
                'F1-Score': '{:.2%}',
                'Suporte': '{:,.0f}'
            }),
            width='stretch'
        )

        # Exibe a acurácia global isolada conforme as boas práticas estatísticas
        st.markdown(f"**Acurácia Global do Modelo:** {acuracia_global:.2%}".replace('.', ','))        

        c1, c2, c3 = st.columns([1, 1, 1.4])

        with c3:
            report_df, acuracia_global = metrics_report(y_test, y_pred)

            csv_report = report_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Baixar Relatório de Métricas (CSV)",
                data=csv_report,
                file_name="relatorio_desempenho_modelo.csv",
                mime="text/csv",
                width='content'
            )

st.subheader("💡 Diagnóstico do Modelo")
with st.expander('Clique para ver detalhes do diagnóstico e recomendações de melhoria'):
    st.markdown("Relatório detalhado das métricas de desempenho por classe, incluindo precisão, recall e F1-Score.")
    st.info(
        "**Análise de Desempenho por Categoria:**\n\n"
        "• O modelo apresenta excelente desempenho geral em classes majoritárias (como *Inaceitável* e *Aceitável*).\n\n"
        "• Com a aplicação do balanceamento no conjunto de treino (`RandomOverSampler`), mitigamos o viés anterior nas classes minoritárias.\n\n"
        "• Acompanhe sempre o **Recall** e o **F1-Score** por classe para garantir que nenhuma categoria seja ignorada."
    )