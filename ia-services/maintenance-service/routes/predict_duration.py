from fastapi import APIRouter
from model_loader import model, scaler, maintenance_columns
from models.duration_input import DurationInput
import pandas as pd

router = APIRouter()

@router.post("/predict/duration")
def predict_duration(data: DurationInput):
    input_dict = data.dict()
    df = pd.DataFrame([input_dict])

    # One-hot encoding
    df = pd.get_dummies(df)

    # Ajouter les colonnes manquantes
    for col in maintenance_columns:
        if col not in df.columns:
            df[col] = 0

    # Réordonner
    df = df[maintenance_columns]

    # Normalisation
    df_scaled = scaler.transform(df)

    # Prédiction
    prediction = model.predict(df_scaled)[0]
    return {"predicted_duration_days": round(float(prediction), 2)}
