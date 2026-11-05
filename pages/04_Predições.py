import streamlit as st
import pandas as pd
from src.data_loader import load_data
from src.model_training import model
from src.charts import metrics_report

# Configuração da página
st.set_page_config(
    page_title="Fazer Previsões - Classificação de Veículos",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Simulador de Previsão de Qualidade")
st.markdown("Preencha os atributos técnicos do veículo para estimar a sua classe de qualidade e visualizar a distribuição de probabilidades do modelo Naive Bayes.")

# Carrega os dados e obtém o modelo treinado e o encoder
df = load_data()
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

# ─── ORDEM LÓGICA DAS OPÇÕES ─────────────────────────────────────────────────
# Listas fixas (do menor para o maior) em vez de df[col].unique(),
# que devolveria os valores embaralhados, na ordem do CSV
ordem_buying_maint = ['low', 'med', 'high', 'vhigh'] 
ordem_doors = ['2', '3', '4', '5more']       
ordem_people = ['2', '4', 'more']
ordem_lug_boot = ['small', 'med', 'big']
ordem_safety = ['low', 'med', 'high']

# ─── FORMULÁRIO ──────────────────────────────────────────────────────────────

with st.container(border=True):
    st.subheader("🚗 Nova Previsão")
    col_esq, col_dir = st.columns([1, 1])

    with col_esq:
        # st.form agrupa os campos: a página só é reprocessada ao clicar no botão
        with st.form('previsão'):
            preco      = st.selectbox('Preço',      [traducao['buying'][v] for v in ordem_buying_maint])            
            manutencao = st.selectbox('Manutenção', [traducao['maint'][v]  for v in ordem_buying_maint])
            pessoas = st.selectbox('Capacidade de Passageiros', [traducao['persons'][v] for v in ordem_people])
            portas  = st.selectbox('Portas',                    [traducao['doors'][v]   for v in ordem_doors])            
            porta_malas = st.selectbox('Porta Malas', [traducao['lug_boot'][v] for v in ordem_lug_boot])
            segurança   = st.selectbox('Segurança',   [traducao['safety'][v]   for v in ordem_safety])

            # Botão de envio do formulário: retorna True quando clicado
            processar = st.form_submit_button('🔮 Fazer Previsão')
    
    with col_dir:        
        # ─── RESULTADO ───────────────────────────────────────────────────────  
        # Fica FORA do st.form: o resultado é exibido abaixo do formulário

        if processar:
            # Valores escolhidos, ainda em português, na mesma ordem das colunas do CSV
            input_features_traduzidos = [preco, manutencao, portas, pessoas, porta_malas, segurança]
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
            probabilidades = modelo.predict_proba(input_encoder)[0]

            # Nomes originais das classes, na mesma ordem dos códigos 0, 1, 2, 3
            # cat.categories → ['acc', 'good', 'unacc', 'vgood'] (ordem alfabética)
            categorias = df['class'].astype('category').cat.categories
                
            # Converte o código de volta para o nome da classe em inglês
            # cat.categories → ['acc', 'good', 'unacc', 'vgood'] (ordem alfabética)
            previsao_original = categorias[predict_encoded][0]

            # Traduz para português; .get usa o valor original se não achar tradução
            previsao_traduzida = traducao['class'].get(previsao_original, previsao_original)

            # Tabela de probabilidades: modelo.classes_ são os códigos das classes
            # na mesma ordem das colunas de predict_proba
            # sort_values → classe mais provável primeiro
            resultado = pd.DataFrame({
                'Classe':[traducao['class'][c] for c in categorias[modelo.classes_]],
                'Probabilidade': probabilidades,
            }).sort_values(by='Probabilidade', ascending=False)
        
            # Card com o resultado e as probabilidades
            with st.container(border=True):
                st.subheader(f"🔮 Resultado da Classificação")

                # Cada classe aparece em um card de cor diferente
                if previsao_traduzida == 'Inaceitável':
                    st.error(f'Resultado da previsão: 🔴 {previsao_traduzida}')      # Vermelho
                elif previsao_traduzida == 'Aceitável':
                    st.warning(f'Resultado da previsão: 🟡 {previsao_traduzida}')    # Amarelo
                elif previsao_traduzida == 'Bom':
                    st.success(f'Resultado da previsão: 🟢 {previsao_traduzida}')    # Verde
                else:
                    st.info(f'Resultado da previsão: 💎 {previsao_traduzida}')       # Azul (Muito Bom)
                
                st.caption('Classe prevista pelo modelo Naive Bayes')
                
                st.markdown("#### 📈 Probabilidade Estimada por Classe")
                st.markdown("Compreenda como o modelo distribuiu a sua confiança entre as categorias:")

                # Uma linha por classe: nome e barra à esquerda, percentual à direita
                for _, linha in resultado.iterrows():
                    col_l, col_r = st.columns([4, 1])

                    with col_l:
                        st.write(f"**{linha['Classe']}**")
                        # st.progress exige um float entre 0 e 1
                        st.progress(float(linha['Probabilidade']))

                    with col_r:
                        # label_visibility='collapsed' → mostra só o número
                        st.metric(
                            label='Confiança', 
                            value=f"{linha['Probabilidade']:.1%}",
                            label_visibility='collapsed'
                        )
    
st.caption(f'Algoritmo: Categorical Naive Bayes | As probabilidades do Naive Bayes refletem a verossimilhança calculada a partir'
    'dos atributos e não são necessariamente calibradas.')

# Exemplo de expansor com st.info explicativo para a página de Predições
with st.expander("ℹ️ Detalhes Técnicos e Metodologia da Página de Predições"):
    st.info(
        """
        **📌 Características e Funcionalidades do Módulo de Predições:**

        * **Formulário Dinâmico de Atributos:** Permite a simulação em tempo real informando os 6 atributos do veículo (*Preço de Compra*, *Custo de Manutenção*, *Portas*, *Capacidade de Passageiros*, *Porta-Malas* e *Nível de Segurança*).
        * **Pré-Processamento Consistente:** Transforma os dados de entrada através do `OrdinalEncoder` treinado com os dados de origem, garantindo alinhamento das categorias sem *data leakage*.
        * **Classificação Multiclasse:** Utiliza o algoritmo **Categorical Naive Bayes** treinado com dados balanceados (`RandomOverSampler`) para evitar viés em categorias minoritárias.
        * **Distribuição de Probabilidades:** Exibe não apenas a classe final prevista, mas também as barras de probabilidade/confiança (`predict_proba`) para cada uma das 4 categorias de qualidade (*Inaceitável*, *Aceitável*, *Bom* e *Muito Bom*).
        * **Feedback Visual Contextual:** Destaque em cores e indicadores visuais personalizados de acordo com o resultado da classificação.
        """
    )

