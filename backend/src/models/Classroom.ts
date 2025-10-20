import mongoose, { Schema, Document } from "mongoose";

export interface IClassroom extends Document {
  name: string;
  code: string;
  teacher: mongoose.Schema.Types.ObjectId;
  students: mongoose.Schema.Types.ObjectId[];
  pendingRequests: mongoose.Schema.Types.ObjectId[]; // 👈 NEW
}

const ClassroomSchema = new Schema<IClassroom>({
  name: { type: String, required: true },
  code: { type: String, required: true, unique: true },
  teacher: { type: Schema.Types.ObjectId, ref: "User", required: true },
  students: [{ type: Schema.Types.ObjectId, ref: "User" }],
  pendingRequests: [{ type: Schema.Types.ObjectId, ref: "User" }], // 👈 NEW
});

export default mongoose.model<IClassroom>("Classroom", ClassroomSchema);