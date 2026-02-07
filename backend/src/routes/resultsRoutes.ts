import express, { Request, Response } from "express";
import Result from "../models/Result";

const router = express.Router();

// ================================================
// ✅ POST /api/results → Save a typing session result
// ================================================
router.post("/", async (req: Request, res: Response) => {
  console.log("📩 Received POST /api/results:", req.body); // 🪶 Debug log

  try {
    const {
      userId,
      level,
      wpm,
      accuracy,
      grade,
      sessionType,
      correctCount,
      wrongKeysCount,
      wrongFingersCount,
      skippedCount,
      compositeScore,
      netWpm,
      errorRate
    } = req.body;

    if (!level || wpm === undefined || accuracy === undefined) {
      return res.status(400).json({ message: "Missing required fields" });
    }

    const result = new Result({
      userId,
      level,
      wpm,
      accuracy,
      grade,
      sessionType,
      correctCount: correctCount || 0,
      wrongKeysCount: wrongKeysCount || 0,
      wrongFingersCount: wrongFingersCount || 0,
      skippedCount: skippedCount || 0,
      compositeScore: compositeScore || 0,
      netWpm: netWpm || 0,
      errorRate: errorRate || 0
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
// ✅ GET /api/results → Fetch latest 10 results (Privacy filtered)
// ================================================
router.get("/", async (req: Request, res: Response) => {
  try {
    const { userId } = req.query;
    const filter = userId ? { userId } : {};

    // Use createdAt (from timestamps: true) for more reliable sub-second sorting
    const results = await Result.find(filter).sort({ createdAt: -1 }).limit(10);
    res.json(results);
  } catch (error) {
    console.error("❌ Error fetching results:", error);
    res.status(500).json({ message: "Failed to fetch results" });
  }
});

// ================================================
// ✅ DELETE /api/results/clear → Clear all session history (for testing/cleanup)
// ================================================
router.delete("/clear", async (req: Request, res: Response) => {
  try {
    const result = await Result.deleteMany({});
    console.log(`🗑️ Cleared ${result.deletedCount} session records from database`);
    res.json({
      message: "All session history cleared",
      deletedCount: result.deletedCount
    });
  } catch (error) {
    console.error("❌ Error clearing results:", error);
    res.status(500).json({ message: "Failed to clear results" });
  }
});

export default router;