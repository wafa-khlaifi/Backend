import matplotlib.pyplot as plt
from train_predict_delay import load_data

# 📊 Charger les données comme tu fais dans ton entraînement
data = load_data()

# Vérifier si les données sont bien chargées
if data is not None and not data.empty:
    plt.figure(figsize=(10, 6))
    plt.hist(data['delay_days'], bins=50, color='skyblue', edgecolor='black')
    plt.title("Distribution des délais de maintenance (delay_days)")
    plt.xlabel("Jours de retard")
    plt.ylabel("Nombre d'occurrences")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("❌ Aucune donnée chargée.")
