# Treinamento Local - Análise de Dados e Modelagem Preditiva

## Resumo do Projeto
Este projeto envolve a análise e modelagem preditiva de dados de reservas de hotel. Utilizamos algoritmos de aprendizado de máquina como XGBoost e RandomForest para prever o preço médio por quarto. O conjunto de dados foi dividido em 75% para treinamento e 25% para teste.

## Observações Importantes sobre os Dados e sua Análise

### Preprocessamento dos Dados
- **Remoção de Colunas**: As colunas `Booking_ID` e `booking_status` foram removidas por não contribuírem para a predição.
- **Adição de Colunas**: Foram adicionadas as colunas `no_total_nights` e `no_total_people` para capturar informações combinadas que podem ser relevantes para o modelo.

### Análise Exploratória
- Realizamos uma análise exploratória inicial para entender as características dos dados, identificar padrões e preparar os dados para modelagem.

### Modelagem Preditiva
- **XGBoost**: Utilizamos o GridSearchCV para testar todas as combinações possíveis dos hiperparâmetros em uma grade pré-definida. O melhor resultado obtido foi de 88% de precisão.
- **RandomForest**: Com ajuste fino dos hiperparâmetros, alcançamos uma precisão máxima de 85.85%.

### Fine Tuning dos Modelos
- **Número de Estimadores (n_estimators)**: Este parâmetro define a quantidade de árvores de decisão no modelo. Um número maior pode melhorar a precisão, mas também aumenta o tempo de treinamento.

### Otimização com Optuna
- O Optuna foi utilizado para realizar uma busca inteligente pelo espaço de hiperparâmetros. Ele executa várias tentativas (trials) independentes, cada uma testando uma combinação diferente de hiperparâmetros.

## Resultados e Conclusões
- A abordagem com XGBoost e GridSearchCV mostrou-se mais eficaz, alcançando a maior precisão nos dados de teste.
- A adição das novas colunas `no_total_nights` e `no_total_people` parece ter contribuído positivamente para a performance dos modelos.
