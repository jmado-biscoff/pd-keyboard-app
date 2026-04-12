import express, { Application, Request, Response } from "express";
import mongoose from "mongoose";
import cors from "cors";
import dotenv from "dotenv";
import path from "path";

// Load .env from project root (one directory up from backend/)
dotenv.config({ path: path.join(process.cwd(), "../.env") });

import authRoutes from "./routes/authRoutes";
import studentRoutes from "./routes/studentRoutes";
import teacherRoutes from "./routes/teacherRoutes";
import typingRoutes from "./routes/typingRoutes";
import resultsRoutes from "./routes/resultsRoutes";
import learnRoutes from "./routes/learnRoutes";
import ocrRoutes from "./routes/ocrRoutes";

// Configuration
const app: Application = express();
const PORT = process.env.PORT || 5000;
const MONGO_URI = process.env.MONGO_URI;

if (!MONGO_URI) {
  console.error("MONGO_URI environment variable is required. Set it in .env");
  process.exit(1);
}

// Middleware
const ALLOWED_ORIGINS = process.env.CORS_ORIGINS
  ? process.env.CORS_ORIGINS.split(",").map(o => o.trim())
  : ["http://localhost:5173"];
app.use(cors({ origin: ALLOWED_ORIGINS, credentials: true }));
app.use(express.json());

// Routes
app.use("/api/auth", authRoutes);
app.use("/api/student", studentRoutes);
app.use("/api/teacher", teacherRoutes);
app.use("/api/typing", typingRoutes);
app.use("/api/results", resultsRoutes);
app.use("/api/learn", learnRoutes);
app.use("/api/ocr", ocrRoutes);

// Serve frontend in production
if (process.env.NODE_ENV === "production") {
  app.use(express.static(path.join(__dirname, "../../../frontend/dist")));
  app.get("*", (req: Request, res: Response) => {
    if (!req.path.startsWith("/api")) {
      res.sendFile(path.join(__dirname, "../../../frontend/dist/index.html"));
    }
  });
} else {
  app.get("/", (req: Request, res: Response) => {
    res.send("Typing App Backend is running...");
  });
}

// Database connection
mongoose
  .connect(MONGO_URI)
  .then(() => {
    console.log("MongoDB connected successfully to:", MONGO_URI);
  })
  .catch((err) => {
    console.error("MongoDB connection error:", err.message);
  });

// Start server regardless of MongoDB connection
app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));
