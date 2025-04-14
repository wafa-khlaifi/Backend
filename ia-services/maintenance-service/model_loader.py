import joblib
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(CURRENT_DIR, 'workorder_maintenance_model_rf.pkl'))
scaler = joblib.load(os.path.join(CURRENT_DIR, 'maintenance_scaler.pkl'))
maintenance_columns = joblib.load(os.path.join(CURRENT_DIR, 'maintenance_columns.pkl'))



# Chargement des artefacts pour delay
delay_model = joblib.load(os.path.join(CURRENT_DIR, "delay_model.pkl"))
delay_scaler = joblib.load(os.path.join(CURRENT_DIR, "delay_scaler.pkl"))
delay_columns = joblib.load(os.path.join(CURRENT_DIR, "delay_columns.pkl"))
