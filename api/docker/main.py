from fastapi import FastAPI
from time import sleep

app = FastAPI(title='Simple API', version='0.0.1', description='Just a simple API.')

@app.get('/api/v1/predict')
async def predict():
    return { "msg": "Seems to be working..."}

if __name__ == '__main__':
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=3000, log_level="info", reload=True)