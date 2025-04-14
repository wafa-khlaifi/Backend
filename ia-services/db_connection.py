# ia-services/app/db_connection.py
import pyodbc
import pandas as pd

def connect_to_database():
    try:
        # Informations de connexion à la base de données Azure SQL
        conn = pyodbc.connect(
            'DRIVER={ODBC Driver 17 for SQL Server};'
            'SERVER=maxgps.smartech-tn.com;'  # Remplace par ton serveur
            'DATABASE=demo7613;'                    # Nom de la base de données
            'UID=smguest;'                          # Nom d'utilisateur
            'PWD=smguest;'                 # Mot de passe
        )
        print("Connexion réussie à la base de données !")
        return conn

    except Exception as e:
        print(f"Erreur lors de la connexion à la base de données : {e}")
        return None

# Test de la connexion
conn = connect_to_database()

