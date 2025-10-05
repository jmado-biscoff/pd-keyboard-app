// backend/server.js
import express from "express";
import path from "path";
import { fileURLToPath } from "url";

const app = express();
const PORT = 5000;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Serve frontend build
const frontendDistPath = path.join(__dirname, "../frontend/dist");

app.use(express.static(frontendDistPath));

// API example
app.get("/api/hello", (req, res) => {
  res.json({ message: "Hello from backend!" });
});

// React Router fallback
app.use((req, res) => {
  res.sendFile(path.join(frontendDistPath, "index.html"));
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
