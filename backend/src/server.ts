import express, { Application, Request, Response } from "express";
import mongoose from "mongoose";
import cors from "cors";
import dotenv from "dotenv";
import path from "path";

dotenv.config({ path: path.resolve(process.cwd(), ".env") });

import authRoutes from "./routes/authRoutes";
import studentRoutes from "./routes/studentRoutes";
import teacherRoutes from "./routes/teacherRoutes";
import typingRoutes from "./routes/typingRoutes";
import detectionRouter from "./routes/detectionRoutes";
import resultsRoutes from "./routes/resultsRoutes";

// Configuration
const app: Application = express();
const PORT = process.env.PORT || 5000;
const MONGO_URI = process.env.MONGO_URI || "mongodb+srv://admin:pd-keyboard-app@cluster0.wuypmob.mongodb.net/";

// Middleware
app.use(cors({ origin: ["http://localhost:5173"], credentials: true }));
app.use(express.json());

// Routes
app.use("/api/auth", authRoutes);
app.use("/api/student", studentRoutes);
app.use("/api/teacher", teacherRoutes);
app.use("/api/typing", typingRoutes);
app.use("/api/detect", detectionRouter);
app.use("/api/results", resultsRoutes);
app.use("/api/auth/teacher", teacherRoutes);
app.use("/api/auth/student", studentRoutes);

app.get("/", (req: Request, res: Response) => {
  res.send("Typing App Backend is running...");
});

// Database + server start
mongoose
  .connect(MONGO_URI)
  .then(() => {
    console.log("MongoDB connected successfully to:", MONGO_URI);
    app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));
  })
  .catch((err) => {
    console.error("MongoDB connection error:", err.message);
  });
