import express from "express";
import { verifyToken, AuthRequest } from "../../middleware/authMiddleware";
import Classroom from "../models/Classroom";

const studentRoutes = express.Router();

/**
 * @route GET /api/student/dashboard
 * @desc Protected route for student dashboard
 */
studentRoutes.get("/dashboard", verifyToken, (req: AuthRequest, res) => {
  const user = req.user as any;
  if (user?.role !== "student") {
    res.status(403).json({ message: "Access denied. Students only." });
    return;
  }

  res.json({
    message: `🎓 Welcome, ${user.email}!`,
    info: "This is your student dashboard.",
  });
});

/**
 * @route POST /api/student/join-classroom
 * @desc Student requests to join a classroom (teacher must approve)
 */
studentRoutes.post(
  "/join-classroom",
  verifyToken,
  async (req: AuthRequest, res) => {
    try {
      const { code } = req.body;
      const user = req.user as any;

      // 🧱 Only allow students
      if (user?.role !== "student") {
        return res
          .status(403)
          .json({ message: "Access denied. Students only." });
      }

      // 🔹 Check if the student is already in any classroom
      const existingClass = await Classroom.findOne({ students: user.id });
      if (existingClass) {
        return res.status(400).json({
          message: `You're already part of classroom: ${existingClass.name}`,
        });
      }

      // 🔹 Find the classroom by join code
      const classroom = await Classroom.findOne({ code });
      if (!classroom) {
        return res.status(404).json({ message: "Classroom not found" });
      }

      // 🔹 Check if already requested to join
      if (classroom.pendingRequests.includes(user.id)) {
        return res
          .status(400)
          .json({ message: "You already sent a join request" });
      }

      // 🔹 Add to pending requests
      classroom.pendingRequests.push(user.id);
      await classroom.save();

      return res.status(200).json({
        message: "Join request sent! Awaiting teacher approval.",
      });
    } catch (error) {
      console.error("❌ Error joining classroom:", error);
      return res.status(500).json({ message: "Failed to send join request" });
    }
  }
);

/**
 * @route GET /api/student/my-classrooms
 * @desc Returns all classrooms the logged-in student has joined
 */
studentRoutes.get(
  "/my-classrooms",
  verifyToken,
  async (req: AuthRequest, res) => {
    try {
      const user = req.user as any;

      if (user?.role !== "student") {
        return res
          .status(403)
          .json({ message: "Access denied. Students only." });
      }

      // Find classrooms that include this student’s ID
      const classrooms = await Classroom.find({ students: user.id }).select(
        "name code"
      );

      res.status(200).json(classrooms);
    } catch (error) {
      console.error("❌ Error fetching student classrooms:", error);
      res.status(500).json({ message: "Failed to load classrooms" });
    }
  }
);

export default studentRoutes;