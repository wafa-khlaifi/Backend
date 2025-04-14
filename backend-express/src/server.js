const express = require("express");
const app = express();
const PORT = 3000;

// 🧠 IMPORTANT : corriger le chemin s’il est faux
const priorityRoutes = require("./routes/priorityRoutes"); // ← assure-toi que ce fichier existe

app.use(express.json());

// ✅ Monte bien les routes ici
app.use("/api", priorityRoutes);

// 👇 Ajoute ceci pour debug
app._router.stack
  .filter((r) => r.route)
  .forEach((r) => console.log("🧩 Route enregistrée :", r.route.path));

app.listen(PORT, () => {
  console.log(`🚀 Serveur Express démarré sur http://localhost:${PORT}`);
});
