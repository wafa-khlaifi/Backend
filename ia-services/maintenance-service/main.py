from fastapi import FastAPI
from routes import predict_duration,predict_delay

app = FastAPI()
app.include_router(predict_duration.router)
app.include_router(predict_delay.router)
