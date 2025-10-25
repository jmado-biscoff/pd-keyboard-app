import express, { Request, Response } from "express";
import Result from "../models/Result";

const router = express.Router();

// ================================================
// ✅ POST /api/results → Save a typing session result
// ================================================
router.post("/", async (req: Request, res: Response) => {
  console.log("📩 Received POST /api/results:", req.body); // 🪶 Debug log

  try {
    const { userId, level, wpm, accuracy, grade, sessionType } = req.body;

    if (!level || !wpm || !accuracy) {
      return res.status(400).json({ message: "Missing required fields" });
    }

    const result = new Result({
      userId,
      level,
      wpm,
      accuracy,
      grade,
      sessionType,
    });

    await result.save();
    console.log("✅ Result saved:", result);

    res.status(201).json(result);
  } catch (error) {
    console.error("❌ Error saving result:", error);
    res.status(500).json({ message: "Failed to save result" });
  }
});

// ================================================
// ✅ GET /api/results → Fetch latest 10 results
// ================================================
router.get("/", async (req: Request, res: Response) => {
  try {
    const results = await Result.find().sort({ date: -1 }).limit(10);
    res.json(results);
  } catch (error) {
    console.error("❌ Error fetching results:", error);
    res.status(500).json({ message: "Failed to fetch results" });
  }
});

export default router;