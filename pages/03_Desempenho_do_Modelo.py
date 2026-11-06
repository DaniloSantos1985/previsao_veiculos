# ═════════════════════════════════════════════════════════════════════════════
# 03_Desempenho_do_Modelo.py — Página de avaliação do classificador
#
# Função: apresentar a acurácia global, a matriz de confusão, o relatório
# de métricas por classe (com opção de download) e um diagnóstico do modelo.
# ═════════════════════════════════════════════════════════════════════════════

# ─── IMPORTAÇÕES ─────────────────────────────────────────────────────────────

import streamlit as st                                      # Framework para criação da interface web
import pandas as pd                                         # (não utilizado nesta página)
from sklearn.metrics import accuracy_score                  # (não utilizado: a acurácia vem de metrics_report)
from src.data_loader import load_data                       # Lê o CSV de veículos
from src.model_training import model                        # Treina o modelo e devolve encoder, modelo e previsões de teste
from src.charts import matriz_confusao, metrics_report      # matriz_confusao → gráfico da matriz
                                                            # metrics_report → relatório por classe + acurácia global

# ─── CONFIGURAÇÃO DA PÁGINA ──────────────────────────────────────────────────

# Título da aba, ícone e layout em tela cheia
# Deve ser o primeiro comando Streamlit do arquivo
st.set_page_config(
    page_title="Desempenho do Modelo - Classificação de Veículos",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Desempenho e Avaliação do Modelo")
st.markdown("Análise estatística detalhada do classificador Naive Bayes, incluindo métricas por classe, matriz de confusão e validação.")

# ─── DADOS, MODELO E MÉTRICAS ────────────────────────────────────────────────

# Lê o CSV original e obtém o modelo treinado
# model() devolve: encoder, modelo, y_test, y_pred (nessa ordem)
# y_test → classes reais do conjunto de teste | y_pred → classes previstas pelo modelo
df = load_data()
encoder, modelo, y_test, y_pred = model(df)

# Gera o relatório por classe e a acurácia global, separados
# A acurácia fica fora da tabela porque é uma métrica global, não de cada classe
# Esta versão de metrics_report devolve DUAS coisas (tabela e acurácia)
report_df, acuracia_global = metrics_report(y_test, y_pred)

# ─── SEÇÃO 1: MÉTRICAS GLOBAIS DE DESTAQUE ───────────────────────────────────

col1, col2, col3 = st.columns(3)

with col1:
    # :.2% → formata como porcentagem com 2 casas decimais
    # .replace('.', ',') → troca o ponto decimal pela vírgula (padrão brasileiro)
    st.metric(
        label='Acurácia Global',
        value=f'{acuracia_global:.2%}'.replace('.', ',')
    )

with col2:
    # Quantidade de registros do conjunto de teste (e não do dataset inteiro)
    st.metric(
        label='Total de Registros de Testes',
        value=f'{len(y_test)}'
    )

with col3:
    # Métrica Categorical Naive Bayes
    st.metric(
        label='Algoritmo Utilizado',
        value='Categorial Naive Bayes'
    )

# ─── SEÇÃO 2: MATRIZ DE CONFUSÃO E RELATÓRIO ─────────────────────────────────

with st.container(border=True):
    # Duas colunas de largura igual: matriz à esquerda, relatório à direita
    col_esq, col_dir = st.columns([1, 1])

    with col_esq:
        st.subheader("📊 Matriz de Confusão")
        st.markdown("Comparação entre as classes reais e as previsões do modelo no conjunto de teste.")
        # Função de src/charts.py: linhas = classe real, colunas = classe prevista
        matriz_confusao(y_test, y_pred)

    with col_dir:
        st.subheader("📋 Relatório de Métricas por Classe")
        st.markdown("A tabela abaixo apresenta as métricas detalhadas de desempenho do modelo para cada classe, traduzidas para português.")

        # style.format → formata cada coluna da tabela
        # '{:.2%}' → porcentagem com 2 casas | '{:,.0f}' → número inteiro com separador de milhar
        st.dataframe(
            report_df.style.format({
                'Precisão': '{:.2%}',
                'Recall': '{:.2%}',
                'F1-Score': '{:.2%}',
                'Suporte': '{:,.0f}'
            }),
            width='stretch'
        )

        # Acurácia global isolada, abaixo da tabela
        st.markdown(f"**Acurácia Global do Modelo:** {acuracia_global:.2%}".replace('.', ','))

        # Três colunas só para posicionar o botão à direita:
        # c1 e c2 ficam vazias e c3 (a mais larga) recebe o botão
        c1, c2, c3 = st.columns([1, 1, 1.4])

        with c3:            
            # Converte a tabela em CSV (em memória) e codifica em bytes                        
            csv_report = report_df.to_csv(index=True).encode('utf-8')

            # Botão de download do relatório
            # width='content' → o botão tem só a largura do texto
            st.download_button(
                label="📥 Baixar Relatório de Métricas (CSV)",
                data=csv_report,
                file_name="relatorio_desempenho_modelo.csv",
                mime="text/csv",
                width='content'
            )

# ─── SEÇÃO 3: DIAGNÓSTICO ────────────────────────────────────────────────────

st.subheader("💡 Diagnóstico do Modelo")

# expander → seção recolhível, fechada por padrão
with st.expander('Clique para ver detalhes do diagnóstico e recomendações de melhoria'):
    st.markdown("Relatório detalhado das métricas de desempenho por classe, incluindo precisão, recall e F1-Score.")    
    st.info(
        "**Análise de Desempenho por Categoria:**\n\n"
        "• O modelo apresenta excelente desempenho geral em classes majoritárias (como *Inaceitável* e *Aceitável*).\n\n"
        "• Com a aplicação do balanceamento no conjunto de treino (`RandomOverSampler`), mitigamos o viés anterior nas classes minoritárias.\n\n"
        "• Acompanhe sempre o **Recall** e o **F1-Score** por classe para garantir que nenhuma categoria seja ignorada."
    )