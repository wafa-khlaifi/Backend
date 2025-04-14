from fastapi import APIRouter
from model_loader import model, scaler, priority_columns
from models.priority_input import PriorityInput
import pandas as pd

router = APIRouter()

@router.post("/priority")
def predict_priority(data: PriorityInput):
    input_dict = data.dict()
    df = pd.DataFrame([input_dict])

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    # Ajouter les colonnes manquantes avec 0
    for col in priority_columns:
        if col not in df.columns:
            df[col] = 0

    # Garder uniquement les colonnes attendues, dans le bon ordre
    df = df[priority_columns]

    # Normalisation
    df_scaled = scaler.transform(df)

    # Prédiction
    prediction = model.predict(df_scaled)[0]
    return {"predicted_calcpriority": round(float(prediction), 2)}
