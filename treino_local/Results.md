# Observações importantes sobre os dados e sua análise
Random forest 500 estimadores, 30% p/ teste 70% p/treinamento = 0.85978 c/100 estimadores deu 0.8584
Xgboost '' '' '' = 0.86557, c/ 100 estimadores deu 0.83965, c/ 1000 estimadores deu 0.86795

### Remoção da coluna ID e Status.
### Análise das colunas: 

booking_status: Como é um resultado da reserva (cancelada ou não), não deve ser usada como uma feature preditiva.

Em relação ao fine tunning, o nr_estimadores é referente a quantidade de arvores de decisão