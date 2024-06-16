import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('modified_hotel_reservations.csv')

# Supondo que 'no_of_adults' seja uma coluna do seu dataframe 'df'
df['no_of_adults'].hist(bins=20)
plt.title('Distribuição do Número de Adultos')
plt.xlabel('Número de Adultos')
plt.ylabel('Frequência')
plt.show()

import seaborn as sns

# Supondo que 'no_of_adults' seja uma feature numérica e 'label_avg_price_per_room' o label
sns.boxplot(x='label_avg_price_per_room', y='no_of_adults', data=df)
plt.title('Relação entre Número de Adultos e Faixa de Preço')
plt.xlabel('Faixa de Preço')
plt.ylabel('Número de Adultos')
plt.show()
