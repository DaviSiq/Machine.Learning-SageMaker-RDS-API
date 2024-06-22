from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
import pandas as pd
import requests

app = FastAPI(title='Hotel Reservation Prediction API', version='0.0.1', description='API para predição de reservas de hotel.')

# Definir o caminho absoluto do modelo
#model_path = os.path.join(os.path.dirname(__file__), 'model.joblib')

# Carregar o modelo ao iniciar a aplicação
model = joblib.load('python/models/model.joblib')

class Reservation(BaseModel):
    Booking_ID: str
    no_of_adults: int
    no_of_children: int
    no_of_weekend_nights: int
    no_of_week_nights: int
    type_of_meal_plan: str
    required_car_parking_space: int
    room_type_reserved: str
    lead_time: int
    arrival_year: int
    arrival_month: int
    arrival_date: int
    market_segment_type: str
    repeated_guest: int
    no_of_previous_cancellations: int
    no_of_previous_bookings_not_canceled: int
    avg_price_per_room: float
    no_of_special_requests: int
    booking_status: str

@app.post('/api/v1/predict')
async def predict(reservation: Reservation):
    try:
        # Criar o DataFrame a partir dos dados de entrada
        data_dict = {
            "Booking_ID": [reservation.Booking_ID],
            "no_of_adults": [reservation.no_of_adults],
            "no_of_children": [reservation.no_of_children],
            "no_of_weekend_nights": [reservation.no_of_weekend_nights],
            "no_of_week_nights": [reservation.no_of_week_nights],
            "type_of_meal_plan": [reservation.type_of_meal_plan],
            "required_car_parking_space": [reservation.required_car_parking_space],
            "room_type_reserved": [reservation.room_type_reserved],
            "lead_time": [reservation.lead_time],
            "arrival_year": [reservation.arrival_year],
            "arrival_month": [reservation.arrival_month],
            "arrival_date": [reservation.arrival_date],
            "market_segment_type": [reservation.market_segment_type],
            "repeated_guest": [reservation.repeated_guest],
            "no_of_previous_cancellations": [reservation.no_of_previous_cancellations],
            "no_of_previous_bookings_not_canceled": [reservation.no_of_previous_bookings_not_canceled],
            "avg_price_per_room": [reservation.avg_price_per_room],
            "no_of_special_requests": [reservation.no_of_special_requests],
            "booking_status": [reservation.booking_status]
        }

        data = pd.DataFrame(data_dict)

        # Adicionar as colunas derivadas
        data['no_total_people'] = data['no_of_adults'] + data['no_of_children']
        data['no_total_nights'] = data['no_of_weekend_nights'] + data['no_of_week_nights']

        # Prever com o modelo
        prediction = model.predict(data)
        result = {'result': int(prediction[0] + 1)}  # Ajustando o rótulo de volta ao intervalo 1-3
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get('/api/v1/test')
async def test():
    return {"message": "API is working"}

# Função para testar a API com um script Python
def test_api():
    url = "http://127.0.0.1:8000/api/v1/predict"  # Mudança de porta
    data = {
        "Booking_ID": "INN00002",
        "no_of_adults": 2,
        "no_of_children": 0,
        "no_of_weekend_nights": 2,
        "no_of_week_nights": 3,
        "type_of_meal_plan": "Not Selected",
        "required_car_parking_space": 0,
        "room_type_reserved": "Room_Type 1",
        "lead_time": 5,
        "arrival_year": 2018,
        "arrival_month": 11,
        "arrival_date": 6,
        "market_segment_type": "Online",
        "repeated_guest": 0,
        "no_of_previous_cancellations": 0,
        "no_of_previous_bookings_not_canceled": 1,
        "avg_price_per_room": 0.0,
        "no_of_special_requests": 1,
        "booking_status": "Not_Canceled"
    }


    response = requests.post(url, json=data)
    print(response.json())

if __name__ == '__main__':
    import uvicorn
    from multiprocessing import Process
    import time

    # Verificar se a porta está disponível antes de iniciar o servidor
    def is_port_in_use(port):
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('127.0.0.1', port)) == 0

    port = 8000
    if not is_port_in_use(port):
        # Iniciar o servidor FastAPI em um processo separado
        server_process = Process(target=uvicorn.run, args=("main:app",), kwargs={"host": "127.0.0.1", "port": port, "log_level": "info"})
        server_process.start()

        # Aguardar o servidor iniciar
        time.sleep(5)

        # Testar a API
        test_api()

        # Encerrar o servidor após o teste
        server_process.terminate()
        server_process.join()
    else:
        print(f"A porta {port} já está em uso.")
