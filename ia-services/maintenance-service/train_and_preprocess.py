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

# 1. Load data with enriched features
def load_data():
    conn = connect_to_database()
    if conn:
        try:
            query = """
                SELECT
                    w.assetnum,
                    w.location,
                    w.worktype,
                    w.schedstart,
                    w.actstart,
                    w.actfinish,
                    pm.frequency,
                    lt.regularhrs,
                    mt.linecost,
                    DATEDIFF(DAY, w.schedstart, w.actstart) AS delay_days,
                    DATEDIFF(DAY, w.actstart, w.actfinish) AS duration_days
                FROM workorder w
                LEFT JOIN pm ON w.pmnum = pm.pmnum
                LEFT JOIN labtrans lt ON lt.refwo = w.wonum
                LEFT JOIN matusetrans mt ON mt.refwo = w.wonum
                WHERE
                    w.worktype = 'PM'
                    AND w.status IN ('COMP', 'CLOSE')
                    AND w.schedstart IS NOT NULL
                    AND w.actstart IS NOT NULL
                    AND w.actfinish IS NOT NULL
                    AND DATEDIFF(DAY, w.actstart, w.actfinish) > 0
            """
            data = pd.read_sql(query, conn)
            data.columns = data.columns.str.strip()
            return data
        except Exception as e:
            print(f"Erreur SQL : {e}")
        finally:
            conn.close()
    return None

# 2. Preprocess data
def preprocess_data(data):
    features = ['delay_days', 'assetnum', 'location', 'worktype', 'frequency', 'regularhrs', 'linecost']
    data = data[features + ['duration_days']].copy()

    # Fill missing numeric values with mean
    num_cols = ['delay_days', 'frequency', 'regularhrs', 'linecost']
    data[num_cols] = data[num_cols].fillna(data[num_cols].mean())

    # One-hot encode categorical
    data = pd.get_dummies(data, columns=['assetnum', 'location', 'worktype'], drop_first=True)

    # Save target
    y = data['duration_days']
    X = data.drop(columns=['duration_days'])

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return pd.DataFrame(X_scaled, columns=X.columns), y, scaler

# 3. Train model
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

# 4. Save model, scaler, and columns
def save_all(model, scaler, columns):
    os.makedirs(CURRENT_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(CURRENT_DIR, 'workorder_maintenance_model_rf.pkl'))
    joblib.dump(scaler, os.path.join(CURRENT_DIR, 'maintenance_scaler.pkl'))
    joblib.dump(columns, os.path.join(CURRENT_DIR, 'maintenance_columns.pkl'))
    print("✅ Modèle et préprocesseurs sauvegardés.")

# 5. Main entry point
def main():
    data = load_data()
    if data is not None and not data.empty:
        data = data.dropna(subset=['delay_days', 'duration_days'])
        X, y, scaler = preprocess_data(data)
        model = train_model(X, y)
        save_all(model, scaler, list(X.columns))
    else:
        print("❌ Aucune donnée disponible pour l'entraînement.")

if __name__ == "__main__":
    main()
