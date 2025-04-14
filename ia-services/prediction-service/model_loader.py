import xgboost as xgb
import joblib

# Chargement du modèle
model = xgb.XGBRegressor()
model.load_model("workorder_priority_model.json")

# Chargement du scaler
scaler = joblib.load("priority_scaler.pkl")

# Chargement des colonnes utilisées à l'entraînement
priority_columns = joblib.load("priority_columns.pkl")
