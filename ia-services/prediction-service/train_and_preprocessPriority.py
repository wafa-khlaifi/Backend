# ia-services/train_and_preprocessPriority.py
import os
import sys
import joblib
import pandas as pd
import xgboost as xgb
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from db_connection import connect_to_database
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(PROJECT_ROOT)

def preprocess_data(data):
    features = ['estlabcost', 'estmatcost', 'worktype', 'status', 'estdur']
    data = data[features]
    data.fillna(data.select_dtypes(include=['number']).mean(), inplace=True)
    data = pd.get_dummies(data, drop_first=True)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return pd.DataFrame(data_scaled, columns=data.columns), scaler

def load_data():
    conn = connect_to_database()
    if conn:
        try:
            query = """
                SELECT wonum, estlabcost, estmatcost, worktype, status, estdur, calcpriority
                FROM [dbo].[workorder]
            """
            data = pd.read_sql(query, conn)
            return data
        except Exception as e:
            print(f"Erreur lors de la récupération des données : {e}")
        finally:
            conn.close()
    else:
        print("Connexion échouée.")
        return None

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
   

 # Calcul des différentes métriques de performance
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Affichage des résultats dans la console
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R² (Coefficient de détermination): {r2}")
    return model
def save_all(model, scaler, columns):
    # Utilisation du répertoire actuel où se trouve le fichier d'entraînement
    output_dir = CURRENT_DIR  # Répertoire où est situé le fichier d'entraînement
    os.makedirs(output_dir, exist_ok=True)  # Crée le dossier si nécessaire

    # Sauvegarder le modèle, scaler et colonnes dans le même répertoire
    model.save_model(os.path.join(output_dir, 'workorder_priority_model.json'))  # Modèle
    joblib.dump(scaler, os.path.join(output_dir, 'priority_scaler.pkl'))  # Scaler
    joblib.dump(columns, os.path.join(output_dir, 'priority_columns.pkl'))  # Colonnes

    print("✅ Modèle, scaler et colonnes sauvegardés dans le répertoire d'entraînement.")

def main():
    data = load_data()
    if data is not None:
        data = data[data['calcpriority'].notnull()]
        data = data[~data['calcpriority'].isin([float('inf'), float('-inf')])]
        if data.empty:
            print("Aucune donnée valide pour entraînement.")
            return
        y = data['calcpriority'].astype('float32')
        X, scaler = preprocess_data(data.drop(columns=['calcpriority']))
        model = train_model(X, y)
        save_all(model, scaler, list(X.columns))

if __name__ == "__main__":
    main()
