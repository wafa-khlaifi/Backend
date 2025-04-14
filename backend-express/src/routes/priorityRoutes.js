const express = require("express");
const router = express.Router();
const { getPredictedPriority } = require("../controllers/priorityController");

router.post("/priority", getPredictedPriority);

module.exports = router;
