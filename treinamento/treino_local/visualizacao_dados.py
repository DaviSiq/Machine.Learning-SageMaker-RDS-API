import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Configurações de estilo do Seaborn
sns.set(style="whitegrid")

df = pd.read_csv('treinamento/data_config/modified_hotel_reservations.csv')

# Cria uma figura e um conjunto de subplots
fig, axes = plt.subplots(3, 1, figsize=(15, 15))  # Ajustado para 3 gráficos

# Histograma do número de adultos com KDE
sns.histplot(df['no_of_adults'], bins=20, kde=True, ax=axes[0])
axes[0].set_title('Distribuição do Número de Adultos')
axes[0].set_xlabel('Número de Adultos')
axes[0].set_ylabel('Frequência')

# Boxplot da relação entre número de adultos e faixa de preço com swarm plot
sns.boxplot(x='label_avg_price_per_room', y='no_of_adults', data=df, ax=axes[1])
sns.swarmplot(x='label_avg_price_per_room', y='no_of_adults', data=df, color='.25', ax=axes[1])
axes[1].set_title('Relação entre Número de Adultos e Faixa de Preço')
axes[1].set_xlabel('Faixa de Preço')
axes[1].set_ylabel('Número de Adultos')

# Correlação entre as variáveis numéricas
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axes[2])
axes[2].set_title('Matriz de Correlação das Variáveis Numéricas')

# Ajustar o layout para evitar sobreposições
plt.tight_layout()

# Mostrar os gráficos
plt.show()
