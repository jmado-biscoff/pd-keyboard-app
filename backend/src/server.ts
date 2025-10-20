//comment
import express, { Application, Request, Response } from "express";
import mongoose from "mongoose";
import cors from "cors";
import dotenv from "dotenv";
import path from "path";

// ✅ Load .env (no matter where server.ts is located)
dotenv.config({ path: path.resolve(process.cwd(), ".env") });


// ✅ Import routes
import authRoutes from "./routes/authRoutes";
import studentRoutes from "./routes/studentRoutes";
import teacherRoutes from "./routes/teacherRoutes";

// ================================================
// CONFIGURATION
// ================================================
const app: Application = express();
const PORT = process.env.PORT || 5000;
const MONGO_URI =
  process.env.MONGO_URI || "mongodb://127.0.0.1:27017/typing_app";

// ================================================
// MIDDLEWARE
// ================================================
app.use(
  cors({
    origin: ["http://localhost:5173"], // frontend
    credentials: true,
  })
);
app.use(express.json());

// ================================================
// ROUTES
// ================================================
app.use("/api/auth", authRoutes);
app.use("/api/student", studentRoutes);
app.use("/api/teacher", teacherRoutes);

// Default route
app.get("/", (req: Request, res: Response) => {
  res.send("✅ Typing App Backend is running...");
});

// ================================================
// DATABASE + SERVER START
// ================================================
mongoose
  .connect(MONGO_URI)
  .then(() => {
    console.log("✅ MongoDB connected successfully to:", MONGO_URI);
    app.listen(PORT, () =>
      console.log(`🚀 Server running on http://localhost:${PORT}`)
    );
  })
  .catch((err) => {
    console.error("❌ MongoDB connection error:", err.message);
  });
