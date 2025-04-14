const axios = require("axios");

exports.getPredictedPriority = async (req, res) => {
  try {
    const fastApiUrl = "http://localhost:8000/predict/priority";
    const response = await axios.post(fastApiUrl, req.body);

    res.status(200).json({
      predictedPriority: response.data.predicted_calcpriority
    });
  } catch (error) {
    console.error("Erreur FastAPI:", error.message);
    res.status(500).json({ error: "Erreur lors de la prédiction IA" });
  }
};
