//comment
import mongoose from "mongoose";

const userSchema = new mongoose.Schema(
  {
    name: { type: String, required: true },
    email: { type: String, required: true, unique: true },
    password: { type: String, required: true },
    role: { type: String, enum: ["student", "teacher"], required: true },
    profilePicture: { type: String, default: "" },
    learningProgress: {
      type: Map,
      of: {
        completed: { type: Boolean, default: false },
        accuracy: { type: Number, default: 0 },
        lastAttemptDate: { type: Date, default: null },
        attempts: { type: Number, default: 0 },
      },
      default: new Map(),
    },
  },
  { timestamps: true }
);

export const User = mongoose.model("User", userSchema);
