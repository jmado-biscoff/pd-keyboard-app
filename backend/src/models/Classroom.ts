import mongoose, { Schema, Document } from "mongoose";

export interface IActiveEvaluation {
  isActive: boolean;
  level: "characters" | "words" | "both";
  proctorTimerMinutes: number;
  activatedAt: Date | null;
  maxAttempts: number;
}

export interface IClassroom extends Document {
  name: string;
  code: string;
  teacher: mongoose.Schema.Types.ObjectId;
  students: mongoose.Schema.Types.ObjectId[];
  pendingRequests: mongoose.Schema.Types.ObjectId[];
  activeEvaluation: IActiveEvaluation;
}

const ClassroomSchema = new Schema<IClassroom>({
  name: { type: String, required: true },
  code: { type: String, required: true, unique: true },
  teacher: { type: Schema.Types.ObjectId, ref: "User", required: true },
  students: [{ type: Schema.Types.ObjectId, ref: "User" }],
  pendingRequests: [{ type: Schema.Types.ObjectId, ref: "User" }],
  activeEvaluation: {
    isActive: { type: Boolean, default: false },
    level: { type: String, enum: ["characters", "words", "both"], default: "characters" },
    proctorTimerMinutes: { type: Number, min: 5, max: 60, default: 30 },
    activatedAt: { type: Date, default: null },
    maxAttempts: { type: Number, default: 3 },
  },
});

export default mongoose.model<IClassroom>("Classroom", ClassroomSchema);