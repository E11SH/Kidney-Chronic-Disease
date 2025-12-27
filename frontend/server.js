import express from "express";
import bodyParser from "body-parser";
import axios from "axios";
import path from "path";
import { fileURLToPath } from 'url';

// ES Module __dirname equivalent
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = 3000;
const API_URL = 'http://localhost:5000';

// Middleware
app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "views"));
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, "public")));

// ============================================================================
// ROUTES
// ============================================================================

// Home page
app.get("/", (req, res) => {
  res.render("home");
});

// API proxy endpoints (to avoid CORS issues)
app.get("/api/models", async (req, res) => {
  try {
    const response = await axios.get(`${API_URL}/api/models`);
    res.json(response.data);
  } catch (error) {
    console.error("Error fetching models:", error.message);
    res.status(500).json({ 
      success: false, 
      error: "Failed to fetch models from backend" 
    });
  }
});

app.post("/api/predict", async (req, res) => {
  try {
    const response = await axios.post(`${API_URL}/api/predict`, req.body);
    res.json(response.data);
  } catch (error) {
    console.error("Error making prediction:", error.message);
    res.status(500).json({ 
      success: false, 
      error: "Failed to make prediction" 
    });
  }
});

app.get("/api/features", async (req, res) => {
  try {
    const response = await axios.get(`${API_URL}/api/features`);
    res.json(response.data);
  } catch (error) {
    console.error("Error fetching features:", error.message);
    res.status(500).json({ 
      success: false, 
      error: "Failed to fetch features" 
    });
  }
});

app.get("/api/model/info/:modelName", async (req, res) => {
  try {
    const { modelName } = req.params;
    const response = await axios.get(`${API_URL}/api/model/info/${modelName}`);
    res.json(response.data);
  } catch (error) {
    console.error("Error fetching model info:", error.message);
    res.status(500).json({ 
      success: false, 
      error: "Failed to fetch model info" 
    });
  }
});

app.get("/api/feature/importance", async (req, res) => {
  try {
    const model = req.query.model || 'random_forest';
    const response = await axios.get(`${API_URL}/api/feature/importance?model=${model}`);
    res.json(response.data);
  } catch (error) {
    console.error("Error fetching feature importance:", error.message);
    res.status(500).json({ 
      success: false, 
      error: "Failed to fetch feature importance" 
    });
  }
});

// Health check
app.get("/health", (req, res) => {
  res.json({ 
    status: "ok", 
    service: "CKD Frontend",
    api_url: API_URL
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).send("Page not found");
});

// Start server
app.listen(port, () => {
  console.log("="

.repeat(80));
  console.log("🚀 CKD INSIGHT FRONTEND SERVER");
  console.log("=".repeat(80));
  console.log(`\n✓ Server running on: http://localhost:${port}`);
  console.log(`✓ API backend at: ${API_URL}`);
  console.log(`\n📝 Available routes:`);
  console.log(`  - GET  /              → Home page`);
  console.log(`  - GET  /health        → Health check`);
  console.log(`  - GET  /api/models    → List models`);
  console.log(`  - POST /api/predict   → Make prediction`);
  console.log("=".repeat(80));
});