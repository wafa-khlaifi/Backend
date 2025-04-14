import os
import sys
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from db_connection import connect_to_database

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(PROJECT_ROOT)

# 1. Load data (excluding frequency, keeping usable fields)
def load_data():
    conn = connect_to_database()
    if conn:
        try:
            query = """
                SELECT
                    w.wonum,
                    w.assetnum,
                    w.location,
                    w.worktype,
                    w.calcpriority,
                    w.schedstart,
                    w.actstart,
                    DATEDIFF(DAY, w.schedstart, w.actstart) AS delay_days
                FROM workorder w
                WHERE
                    w.status IN ('COMP', 'CLOSE')
                    AND w.schedstart IS NOT NULL
                    AND w.actstart IS NOT NULL
                    AND DATEDIFF(DAY, w.schedstart, w.actstart) >= 0
                    AND w.worktype IS NOT NULL
            """
            print("📡 Exécution de la requête SQL en cours...")
            data = pd.read_sql(query, conn)
            print("✅ Données chargées :", len(data), "lignes.")

            data.columns = data.columns.str.strip()
            return data
        except Exception as e:
            print(f"Erreur SQL : {e}")
        finally:
            conn.close()
    return None

# 2. Map worktypes into grouped categories
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

# 3. Preprocess data
def preprocess_data(data):
    data = data.copy()
    data['worktype'] = data['worktype'].apply(map_worktype)

    # Extract weekday from schedstart (optional feature)
    data['sched_weekday'] = pd.to_datetime(data['schedstart']).dt.weekday

    features = ['assetnum', 'location', 'worktype', 'calcpriority', 'sched_weekday']
    data = data[features + ['delay_days']].copy()

    # Fill missing numeric values
    data['calcpriority'] = data['calcpriority'].fillna(data['calcpriority'].mean())

    # One-hot encode categorical features
    data = pd.get_dummies(data, columns=['assetnum', 'location', 'worktype'], drop_first=True)

    y = data['delay_days']
    X = data.drop(columns=['delay_days'])

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return pd.DataFrame(X_scaled, columns=X.columns), y, scaler

# 4. Train model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"R²: {r2_score(y_test, y_pred):.2f}")

    return model

# 5. Save model, scaler, and columns
def save_all(model, scaler, columns):
    os.makedirs(CURRENT_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(CURRENT_DIR, 'delay_model.pkl'))
    joblib.dump(scaler, os.path.join(CURRENT_DIR, 'delay_scaler.pkl'))
    joblib.dump(columns, os.path.join(CURRENT_DIR, 'delay_columns.pkl'))
    print("✅ Modèle et préprocesseurs sauvegardés.")

# 6. Main
def main():
    data = load_data()
    if data is not None and not data.empty:
        X, y, scaler = preprocess_data(data)
        model = train_model(X, y)
        save_all(model, scaler, list(X.columns))
    else:
        print("❌ Aucune donnée disponible pour l'entraînement.")

main()
