import express from "express";
import { verifyToken, AuthRequest } from "../../middleware/authMiddleware";
import Classroom from "../models/Classroom";
import { User } from "../models/User";

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

      // Find classrooms that include this student's ID, populate teacher name
      const classrooms = await Classroom.find({ students: user.id })
        .select("name code teacher")
        .populate("teacher", "name");

      res.status(200).json(classrooms);
    } catch (error) {
      console.error("❌ Error fetching student classrooms:", error);
      res.status(500).json({ message: "Failed to load classrooms" });
    }
  }
);

/**
 * @route GET /api/student/evaluation-status
 * @desc Check if student has an active evaluation in any enrolled classroom
 */
studentRoutes.get(
  "/evaluation-status",
  verifyToken,
  async (req: AuthRequest, res) => {
    try {
      const user = req.user as any;
      if (user?.role !== "student") {
        return res.status(403).json({ message: "Access denied. Students only." });
      }

      // Find classrooms this student is enrolled in
      const classrooms = await Classroom.find({ students: user.id });

      // Check for any active evaluation
      for (const classroom of classrooms) {
        const eval_ = classroom.activeEvaluation;
        if (eval_ && eval_.isActive && eval_.activatedAt) {
          const activatedAt = new Date(eval_.activatedAt).getTime();
          const expiresAt = activatedAt + eval_.proctorTimerMinutes * 60 * 1000;
          const now = Date.now();
          const remainingSeconds = Math.max(0, Math.floor((expiresAt - now) / 1000));

          if (remainingSeconds <= 0) {
            // Evaluation has expired — auto-deactivate
            eval_.isActive = false;
            eval_.activatedAt = null;
            await classroom.save();
            continue;
          }

          return res.json({
            hasActiveEvaluation: true,
            evaluation: {
              classroomId: classroom._id,
              classroomName: classroom.name,
              level: eval_.level,
              proctorTimerMinutes: eval_.proctorTimerMinutes,
              activatedAt: eval_.activatedAt.toISOString(),
              expiresAt: new Date(expiresAt).toISOString(),
              remainingSeconds,
              maxAttempts: eval_.maxAttempts,
              sessionId: eval_.sessionId || null,
            },
          });
        }
      }

      return res.json({ hasActiveEvaluation: false, evaluation: null });
    } catch (error) {
      console.error("Error checking evaluation status:", error);
      res.status(500).json({ message: "Failed to check evaluation status" });
    }
  }
);

/**
 * @route GET /api/student/learning-progress
 * @desc Get student's learning module progress
 */
studentRoutes.get(
  "/learning-progress",
  verifyToken,
  async (req: AuthRequest, res) => {
    try {
      const user = req.user as any;
      if (user?.role !== "student") {
        return res.status(403).json({ message: "Access denied. Students only." });
      }

      const student = await User.findById(user.id);
      if (!student) {
        return res.status(404).json({ message: "User not found" });
      }

      // Convert Map to plain object for JSON response
      const progress = student.learningProgress || new Map();
      const progressObj: Record<string, any> = {};
      progress.forEach((value: any, key: string) => {
        progressObj[key] = value;
      });

      res.status(200).json(progressObj);
    } catch (error) {
      console.error("Error fetching learning progress:", error);
      res.status(500).json({ message: "Failed to fetch learning progress" });
    }
  }
);

/**
 * @route POST /api/student/learning-progress
 * @desc Update progress for a specific learning module
 */
studentRoutes.post(
  "/learning-progress",
  verifyToken,
  async (req: AuthRequest, res) => {
    try {
      const user = req.user as any;
      if (user?.role !== "student") {
        return res.status(403).json({ message: "Access denied. Students only." });
      }

      const { moduleId, completed, accuracy } = req.body;

      if (!moduleId || typeof completed !== "boolean" || typeof accuracy !== "number") {
        return res.status(400).json({ message: "Invalid request body" });
      }

      const student = await User.findById(user.id);
      if (!student) {
        return res.status(404).json({ message: "User not found" });
      }

      // Initialize learningProgress if not exists
      if (!student.learningProgress) {
        student.learningProgress = new Map();
      }

      // Get existing progress for this module or create new
      const existing = student.learningProgress.get(String(moduleId)) || {
        completed: false,
        accuracy: 0,
        lastAttemptDate: null,
        attempts: 0,
      };

      // Update progress
      student.learningProgress.set(String(moduleId), {
        completed,
        accuracy,
        lastAttemptDate: new Date(),
        attempts: existing.attempts + 1,
      });

      await student.save();

      res.status(200).json({ message: "Progress saved successfully" });
    } catch (error) {
      console.error("Error saving learning progress:", error);
      res.status(500).json({ message: "Failed to save learning progress" });
    }
  }
);

/**
 * @route PUT /api/student/learning-progress/reset
 * @desc Reset all learning progress for the student
 */
studentRoutes.put(
  "/learning-progress/reset",
  verifyToken,
  async (req: AuthRequest, res) => {
    try {
      const user = req.user as any;
      if (user?.role !== "student") {
        return res.status(403).json({ message: "Access denied. Students only." });
      }

      const student = await User.findById(user.id);
      if (!student) {
        return res.status(404).json({ message: "User not found" });
      }

      // Clear all learning progress
      student.learningProgress = new Map();
      await student.save();

      res.status(200).json({ message: "Learning progress reset successfully" });
    } catch (error) {
      console.error("Error resetting learning progress:", error);
      res.status(500).json({ message: "Failed to reset learning progress" });
    }
  }
);

export default studentRoutes;