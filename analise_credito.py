import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, confusion_matrix, roc_curve
import seaborn as sns
import requests
from io import StringIO

# Título da aplicação
st.title("Análise de Crédito")
st.markdown("## A análise é fictícia, serve apenas para fins de estudo")

# Sidebar para upload de dados
st.sidebar.header("Upload de Dados")
uploaded_file = st.sidebar.file_uploader("Carregue o arquivo CSV com os dados dos clientes", type=["csv"])

# Botão para carregar dados do GitHub
if st.sidebar.button("Carregar dados do GitHub"):
    url = "https://github.com/daltonffarias/analise_credito/blob/main/dados_credito.csv"
    response = requests.get(url)
    if response.status_code == 200:
        df = pd.read_csv(StringIO(response.text))
        st.session_state['df'] = df
        st.sidebar.success("Dados carregados com sucesso do GitHub!")
    else:
        st.sidebar.error("Erro ao carregar dados do GitHub.")

# Verifica se o DataFrame está na sessão
if 'df' in st.session_state:
    df = st.session_state['df']

    # Exibir dados brutos
    st.subheader("Dados Brutos")
    st.write(df)

    # Processamento de dados
    st.subheader("Processamento de Dados")
    df['idade'] = 2023 - df['ano_nascimento']
    df = df.dropna()
    df = df.drop_duplicates()

    # Exibir dados processados
    st.write("Dados Processados:")
    st.write(df)

    # Validação de dados
    st.subheader("Validação de Dados")
    if df['idade'].min() >= 18 and df['renda'].min() >= 0:
        st.success("Dados validados com sucesso!")
    else:
        st.error("Erro na validação dos dados. Verifique os critérios.")

    # Modelos Analíticos
    st.subheader("Modelos Analíticos")

    # Renda Presumida
    st.markdown("### Renda Presumida")
    X_income = df[['idade', 'educacao', 'experiencia']]
    y_income = df['renda']
    X_train_income, X_test_income, y_train_income, y_test_income = train_test_split(X_income, y_income, test_size=0.2, random_state=42)
    income_model = LinearRegression()
    income_model.fit(X_train_income, y_train_income)
    y_pred_income = income_model.predict(X_test_income)
    mse_income = mean_squared_error(y_test_income, y_pred_income)
    st.write(f"Erro Quadrático Médio (MSE) para Renda Presumida: {mse_income:.2f}")

    # Gráfico de dispersão para Renda Presumida
    fig, ax = plt.subplots()
    ax.scatter(y_test_income, y_pred_income, alpha=0.5)
    ax.plot([y_test_income.min(), y_test_income.max()], [y_test_income.min(), y_test_income.max()], 'k--', lw=2)
    ax.set_xlabel('Renda Real')
    ax.set_ylabel('Renda Prevista')
    ax.set_title('Renda Presumida: Real vs Previsto')
    st.pyplot(fig)

    # Capacidade de Pagamento
    st.markdown("### Capacidade de Pagamento")
    X_payment = df[['renda', 'divida', 'idade']]
    y_payment = df['capacidade_pagamento']
    X_train_payment, X_test_payment, y_train_payment, y_test_payment = train_test_split(X_payment, y_payment, test_size=0.2, random_state=42)
    payment_model = RandomForestClassifier()
    payment_model.fit(X_train_payment, y_train_payment)
    y_pred_payment = payment_model.predict(X_test_payment)
    accuracy_payment = accuracy_score(y_test_payment, y_pred_payment)
    st.write(f"Acurácia para Capacidade de Pagamento: {accuracy_payment:.2f}")

    # Matriz de Confusão para Capacidade de Pagamento
    cm = confusion_matrix(y_test_payment, y_pred_payment)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Previsto')
    ax.set_ylabel('Real')
    ax.set_title('Matriz de Confusão para Capacidade de Pagamento')
    st.pyplot(fig)

    # Risco de Inadimplência
    st.markdown("### Risco de Inadimplência")
    X_risk = df[['renda', 'divida', 'idade', 'historico_credito']]
    y_risk = df['inadimplente']
    X_train_risk, X_test_risk, y_train_risk, y_test_risk = train_test_split(X_risk, y_risk, test_size=0.2, random_state=42)
    risk_model = GradientBoostingClassifier()
    risk_model.fit(X_train_risk, y_train_risk)
    y_pred_risk = risk_model.predict_proba(X_test_risk)[:, 1]
    roc_auc_risk = roc_auc_score(y_test_risk, y_pred_risk)
    st.write(f"AUC-ROC para Risco de Inadimplência: {roc_auc_risk:.2f}")

    # Curva ROC para Risco de Inadimplência
    fpr, tpr, _ = roc_curve(y_test_risk, y_pred_risk)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc_risk:.2f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('Taxa de Falsos Positivos')
    ax.set_ylabel('Taxa de Verdadeiros Positivos')
    ax.set_title('Curva ROC para Risco de Inadimplência')
    ax.legend(loc="lower right")
    st.pyplot(fig)

    # Relatórios e Visualizações
    st.subheader("Relatórios e Visualizações")

    # Gráfico de Desempenho dos Modelos
    st.markdown("### Desempenho dos Modelos")
    performance_metrics = {
        'Renda Presumida': mse_income,
        'Capacidade de Pagamento': accuracy_payment,
        'Risco de Inadimplência': roc_auc_risk
    }
    fig, ax = plt.subplots()
    ax.bar(performance_metrics.keys(), performance_metrics.values())
    ax.set_ylabel("Desempenho")
    ax.set_title("Desempenho dos Modelos")
    st.pyplot(fig)

    # Relatório de Métricas de Risco
    st.markdown("### Relatório de Métricas de Risco")
    st.write(performance_metrics)

elif uploaded_file is not None:
    # Carregar dados
    df = pd.read_csv(uploaded_file)
    st.session_state['df'] = df  # Armazena o DataFrame na sessão

    # Exibir dados brutos
    st.subheader("Dados Brutos")
    st.write(df)

    # Processamento de dados
    st.subheader("Processamento de Dados")
    df['idade'] = 2023 - df['ano_nascimento']
    df = df.dropna()
    df = df.drop_duplicates()

    # Exibir dados processados
    st.write("Dados Processados:")
    st.write(df)

    # Validação de dados
    st.subheader("Validação de Dados")
    if df['idade'].min() >= 18 and df['renda'].min() >= 0:
        st.success("Dados validados com sucesso!")
    else:
        st.error("Erro na validação dos dados. Verifique os critérios.")

    # Modelos Analíticos
    st.subheader("Modelos Analíticos")

    # Renda Presumida
    st.markdown("### Renda Presumida")
    X_income = df[['idade', 'educacao', 'experiencia']]
    y_income = df['renda']
    X_train_income, X_test_income, y_train_income, y_test_income = train_test_split(X_income, y_income, test_size=0.2, random_state=42)
    income_model = LinearRegression()
    income_model.fit(X_train_income, y_train_income)
    y_pred_income = income_model.predict(X_test_income)
    mse_income = mean_squared_error(y_test_income, y_pred_income)
    st.write(f"Erro Quadrático Médio (MSE) para Renda Presumida: {mse_income:.2f}")

    # Gráfico de dispersão para Renda Presumida
    fig, ax = plt.subplots()
    ax.scatter(y_test_income, y_pred_income, alpha=0.5)
    ax.plot([y_test_income.min(), y_test_income.max()], [y_test_income.min(), y_test_income.max()], 'k--', lw=2)
    ax.set_xlabel('Renda Real')
    ax.set_ylabel('Renda Prevista')
    ax.set_title('Renda Presumida: Real vs Previsto')
    st.pyplot(fig)

    # Capacidade de Pagamento
    st.markdown("### Capacidade de Pagamento")
    X_payment = df[['renda', 'divida', 'idade']]
    y_payment = df['capacidade_pagamento']
    X_train_payment, X_test_payment, y_train_payment, y_test_payment = train_test_split(X_payment, y_payment, test_size=0.2, random_state=42)
    payment_model = RandomForestClassifier()
    payment_model.fit(X_train_payment, y_train_payment)
    y_pred_payment = payment_model.predict(X_test_payment)
    accuracy_payment = accuracy_score(y_test_payment, y_pred_payment)
    st.write(f"Acurácia para Capacidade de Pagamento: {accuracy_payment:.2f}")

    # Matriz de Confusão para Capacidade de Pagamento
    cm = confusion_matrix(y_test_payment, y_pred_payment)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Previsto')
    ax.set_ylabel('Real')
    ax.set_title('Matriz de Confusão para Capacidade de Pagamento')
    st.pyplot(fig)

    # Risco de Inadimplência
    st.markdown("### Risco de Inadimplência")
    X_risk = df[['renda', 'divida', 'idade', 'historico_credito']]
    y_risk = df['inadimplente']
    X_train_risk, X_test_risk, y_train_risk, y_test_risk = train_test_split(X_risk, y_risk, test_size=0.2, random_state=42)
    risk_model = GradientBoostingClassifier()
    risk_model.fit(X_train_risk, y_train_risk)
    y_pred_risk = risk_model.predict_proba(X_test_risk)[:, 1]
    roc_auc_risk = roc_auc_score(y_test_risk, y_pred_risk)
    st.write(f"AUC-ROC para Risco de Inadimplência: {roc_auc_risk:.2f}")

    # Curva ROC para Risco de Inadimplência
    fpr, tpr, _ = roc_curve(y_test_risk, y_pred_risk)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc_risk:.2f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('Taxa de Falsos Positivos')
    ax.set_ylabel('Taxa de Verdadeiros Positivos')
    ax.set_title('Curva ROC para Risco de Inadimplência')
    ax.legend(loc="lower right")
    st.pyplot(fig)

    # Relatórios e Visualizações
    st.subheader("Relatórios e Visualizações")

    # Gráfico de Desempenho dos Modelos
    st.markdown("### Desempenho dos Modelos")
    performance_metrics = {
        'Renda Presumida': mse_income,
        'Capacidade de Pagamento': accuracy_payment,
        'Risco de Inadimplência': roc_auc_risk
    }
    fig, ax = plt.subplots()
    ax.bar(performance_metrics.keys(), performance_metrics.values())
    ax.set_ylabel("Desempenho")
    ax.set_title("Desempenho dos Modelos")
    st.pyplot(fig)

    # Relatório de Métricas de Risco
    st.markdown("### Relatório de Métricas de Risco")
    st.write(performance_metrics)

else:
    st.info("Por favor, carregue um arquivo CSV ou clique no botão para carregar os dados do GitHub.")
