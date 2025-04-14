from fastapi import APIRouter
from model_loader import delay_model, delay_scaler, delay_columns
from models.delay_input import DelayInput
import pandas as pd
from datetime import datetime

router = APIRouter()

@router.post("/predict/delay")
def predict_delay(data: DelayInput):
    input_dict = data.dict()

    # Mapping du type de workorder
    def map_worktype(wt):
        if wt in ['PM', 'EV', 'SIG']:
            return 'PM'
        elif wt in ['CM', 'MAJOR', 'MINOR']:
            return 'CM'
        elif wt == 'EM':
            return 'EM'
        elif wt in ['CAL', 'SA', 'A']:
            return 'CALL'
        elif wt == 'CP':
            return 'PROJECT'
        else:
            return 'OTHER'

    input_dict['worktype'] = map_worktype(input_dict['worktype'])
    input_dict['sched_weekday'] = datetime.fromisoformat(input_dict['schedstart']).weekday()
    del input_dict['schedstart']

    df = pd.DataFrame([input_dict])

    # One-hot encoding
    df = pd.get_dummies(df)

    # Ajout des colonnes manquantes
    for col in delay_columns:
        if col not in df.columns:
            df[col] = 0

    # Réordonner les colonnes
    df = df[delay_columns]

    # Normalisation
    df_scaled = delay_scaler.transform(df)

    # Prédiction
    prediction = delay_model.predict(df_scaled)[0]

    return {"predicted_delay_days": round(float(prediction), 2)}
