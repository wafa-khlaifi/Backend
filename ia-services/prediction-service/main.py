# prediction-service/main.py
from fastapi import FastAPI
from routes.priority import router as priority_router

app = FastAPI(
    title="Prediction Service",
    description="Microservice de prédiction de priorité des Work Orders",
    version="1.0.0"
)

app.include_router(priority_router, prefix="/predict",tags=["Prediction"])
