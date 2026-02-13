import mongoose from "mongoose";

const ResultSchema = new mongoose.Schema(
  {
    userId: { type: String, required: false }, // optional until you add auth
    level: { type: Number, required: true },
    wpm: { type: Number, required: true },
    accuracy: { type: Number, required: true },
    grade: { type: String, required: false },
    sessionType: {
      type: String,
      enum: ["practice", "evaluated"],
      default: "practice",
      required: true,
    },
    correctCount: { type: Number, default: 0 },
    wrongKeysCount: { type: Number, default: 0 },
    wrongFingersCount: { type: Number, default: 0 },
    skippedCount: { type: Number, default: 0 },
    compositeScore: { type: Number, default: 0 },
    netWpm: { type: Number, default: 0 },
    errorRate: { type: Number, default: 0 },
    sessionId: { type: String, required: false },
    date: { type: Date, default: Date.now },
  },
  { timestamps: true }
);

// Reuse model if it already exists (hot-reload safe)
export default mongoose.models.Result || mongoose.model("Result", ResultSchema);