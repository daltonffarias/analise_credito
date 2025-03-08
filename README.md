# analise_credito
A análise de crédito é uma das aplicações mais importantes no setor financeiro, pois permite avaliar a capacidade de um cliente de pagar suas dívidas e o risco associado à concessão de crédito. Com o avanço das técnicas de machine learning, é possível automatizar e aprimorar essa análise, utilizando dados históricos e características dos clientes para prever comportamentos futuros.

Neste projeto, desenvolvemos uma aplicação interativa utilizando a biblioteca Streamlit para realizar uma análise de crédito fictícia. O código carrega um conjunto de dados contendo informações sobre clientes, como idade, renda, dívida, histórico de crédito e outros atributos relevantes. A partir desses dados, são aplicados três modelos analíticos:

Renda Presumida:

Utilizamos um modelo de Regressão Linear para prever a renda dos clientes com base em características como idade, nível de educação e experiência profissional.

O desempenho do modelo é avaliado por meio do Erro Quadrático Médio (MSE), que mede a diferença entre os valores reais e os previstos.

Capacidade de Pagamento:

Um modelo de Random Forest Classifier é aplicado para classificar a capacidade de pagamento dos clientes, considerando variáveis como renda, dívida e idade.

A precisão do modelo é avaliada por meio da Acurácia, que indica a proporção de previsões corretas em relação ao total de previsões.

Risco de Inadimplência:

Um modelo de Gradient Boosting Classifier é utilizado para prever o risco de inadimplência dos clientes, com base em características como renda, dívida, idade e histórico de crédito.

O desempenho do modelo é avaliado por meio da AUC-ROC (Área Sob a Curva ROC), que mede a capacidade do modelo de distinguir entre clientes inadimplentes e não inadimplentes.

Além dos modelos, o código inclui visualizações interativas, como gráficos de dispersão, matrizes de confusão e curvas ROC, que ajudam a interpretar os resultados e a entender o desempenho de cada modelo. Essas visualizações são fundamentais para uma análise completa e didática, permitindo que usuários sem conhecimento técnico aprofundado possam compreender as previsões e métricas.
