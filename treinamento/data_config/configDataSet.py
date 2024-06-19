import pandas as pd
import numpy as np
# Carregar o dataset
url = "HotelReservations.csv"
dataset = pd.read_csv(url)

# Criar a nova coluna label_avg_price_per_room
conditions = [
    (dataset['avg_price_per_room'] <= 85),
    (dataset['avg_price_per_room'] > 85) & (dataset['avg_price_per_room'] < 115),
    (dataset['avg_price_per_room'] >= 115)
]
choices = [1, 2, 3]
dataset['label_avg_price_per_room'] = np.select(conditions, choices)

# Excluir a coluna avg_price_per_room
dataset = dataset.drop('avg_price_per_room', axis=1)

# Salvar o dataset alterado
dataset.to_csv('modified_hotel_reservations.csv', index=False)
