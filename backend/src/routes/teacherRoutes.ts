import express from "express";
import { verifyToken, AuthRequest } from "../../middleware/authMiddleware";
import Classroom from "../models/Classroom";
import { User } from "../models/User";
import Result from "../models/Result";
import { generateClassCode } from "../utils/generateClassCode";

const teacherRoutes = express.Router();

/**
 * @route GET /api/teacher/dashboard
 * @desc Protected route for teacher dashboard
 */
teacherRoutes.get("/dashboard", verifyToken, (req: AuthRequest, res) => {
  const user = req.user as any;
  if (user?.role !== "teacher") {
    res.status(403).json({ message: "Access denied. Teachers only." });
    return;
  }

  res.json({
    message: `👩‍🏫 Welcome, ${user.email}!`,
    info: "This is your teacher dashboard.",
  });
});

/**
 * @route POST /api/teacher/create-classroom
 * @desc Create a new classroom with unique code
 */
teacherRoutes.post(
  "/create-classroom",
  verifyToken,
  async (req: AuthRequest, res) => {
    try {
      const user = req.user as any;
      if (user?.role !== "teacher") {
        return res
          .status(403)
          .json({ message: "Access denied. Teachers only." });
      }

      const { name } = req.body;
      if (!name) {
        return res.status(400).json({ message: "Classroom name is required." });
      }

      const code = generateClassCode();

      const classroom = await Classroom.create({
        name,
        code,
        teacher: user.id, // teacher ID from JWT
        students: [],
        pendingRequests: [], // 👈 ensure new field exists when created
      });

      res.status(201).json({
        message: "Classroom created successfully!",
        classroom,
      });
    } catch (error) {
      console.error("Error creating classroom:", error);
      res.status(500).json({ message: "Failed to create classroom." });
    }
  }
);

/**
 * @route GET /api/teacher/my-classrooms
 * @desc Get all classrooms created by the teacher
 */
teacherRoutes.get(
  "/my-classrooms",
  verifyToken,
  async (req: AuthRequest, res) => {
    try {
      const user = req.user as any;
      if (user?.role !== "teacher") {
        return res
          .status(403)
          .json({ message: "Access denied. Teachers only." });
      }

      const classrooms = await Classroom.find({ teacher: user.id });
      res.json(classrooms);
    } catch (error) {
      console.error("Error fetching classrooms:", error);
      res.status(500).json({ message: "Failed to fetch classrooms." });
    }
  }
);

/**
 * @route GET /api/teacher/classroom/:id/students
 * @desc Returns all students enrolled in a specific classroom
 */
teacherRoutes.get(
  "/classroom/:id/students",
  verifyToken,
  async (req: AuthRequest, res) => {
    try {
      const user = req.user as any;

      if (user?.role !== "teacher") {
        return res
          .status(403)
          .json({ message: "Access denied. Teachers only." });
      }

      const { id } = req.params;

      // Find the classroom and populate student details
      const classroom = await Classroom.findById(id).populate(
        "students",
        "name email"
      );

      if (!classroom) {
        return res.status(404).json({ message: "Classroom not found" });
      }

      res.status(200).json({
        name: classroom.name,
        code: classroom.code,
        students: classroom.students,
      });
    } catch (error) {
      console.error("❌ Error fetching classroom students:", error);
      res.status(500).json({ message: "Failed to load classroom students" });
    }
  }
);

/**
 * @route GET /api/teacher/classroom/:id/requests
 * @desc Get pending join requests for a classroom
 */
teacherRoutes.get(
  "/classroom/:id/requests",
  verifyToken,
  async (req: AuthRequest, res) => {
    try {
      const user = req.user as any;

      if (user?.role !== "teacher") {
        return res
          .status(403)
          .json({ message: "Access denied. Teachers only." });
      }

      const classroom = await Classroom.findById(req.params.id).populate(
        "pendingRequests",
        "name email"
      );

      if (!classroom) {
        return res.status(404).json({ message: "Classroom not found" });
      }

      // Ensure this teacher owns the class
      if (String(classroom.teacher) !== String(user.id)) {
        return res.status(403).json({ message: "Not your classroom" });
      }

      res.status(200).json(classroom.pendingRequests);
    } catch (error) {
      console.error("❌ Error fetching join requests:", error);
      res.status(500).json({ message: "Failed to load join requests" });
    }
  }
);

/**
 * @route POST /api/teacher/classroom/:id/approve
 * @desc Approve a student's join request
 */
teacherRoutes.post(
  "/classroom/:id/approve",
  verifyToken,
  async (req: AuthRequest, res) => {
    try {
      const { studentId } = req.body;
      const user = req.user as any;

      if (user?.role !== "teacher") {
        return res
          .status(403)
          .json({ message: "Access denied. Teachers only." });
      }

      const classroom = await Classroom.findById(req.params.id);
      if (!classroom) {
        return res.status(404).json({ message: "Classroom not found" });
      }

      // Ensure this teacher owns the class
      if (String(classroom.teacher) !== String(user.id)) {
        return res.status(403).json({ message: "Not your classroom" });
      }

      // Remove from pendingRequests
      classroom.pendingRequests = classroom.pendingRequests.filter(
        (id: any) => String(id) !== String(studentId)
      );

      // Add to students list (if not already)
      if (!classroom.students.includes(studentId)) {
        classroom.students.push(studentId);
      }

      await classroom.save();
      res.status(200).json({ message: "Student approved successfully" });
    } catch (error) {
      console.error("❌ Error approving student:", error);
      res.status(500).json({ message: "Failed to approve student" });
    }
  }
);

/**
 * @route POST /api/teacher/classroom/:id/reject
 * @desc Reject a student's join request
 */
teacherRoutes.post(
  "/classroom/:id/reject",
  verifyToken,
  async (req: AuthRequest, res) => {
    try {
      const { studentId } = req.body;
      const user = req.user as any;

      if (user?.role !== "teacher") {
        return res
          .status(403)
          .json({ message: "Access denied. Teachers only." });
      }

      const classroom = await Classroom.findById(req.params.id);
      if (!classroom) {
        return res.status(404).json({ message: "Classroom not found" });
      }

      // Ensure this teacher owns the class
      if (String(classroom.teacher) !== String(user.id)) {
        return res.status(403).json({ message: "Not your classroom" });
      }

      // Remove from pendingRequests only
      classroom.pendingRequests = classroom.pendingRequests.filter(
        (id: any) => String(id) !== String(studentId)
      );

      await classroom.save();
      res.status(200).json({ message: "Student rejected successfully" });
    } catch (error) {
      console.error("❌ Error rejecting student:", error);
      res.status(500).json({ message: "Failed to reject student" });
    }
  }
);

/**
 * @route DELETE /api/teacher/classroom/:id
 * @desc Delete a classroom (teacher must be the owner)
 */
teacherRoutes.delete(
  "/classroom/:id",
  verifyToken,
  async (req: AuthRequest, res) => {
    try {
      const user = req.user as any;
      if (user?.role !== "teacher") {
        return res.status(403).json({ message: "Access denied. Teachers only." });
      }

      const classroom = await Classroom.findById(req.params.id);
      if (!classroom) {
        return res.status(404).json({ message: "Classroom not found" });
      }
      if (String(classroom.teacher) !== String(user.id)) {
        return res.status(403).json({ message: "Not your classroom" });
      }

      await Classroom.findByIdAndDelete(req.params.id);
      res.json({ message: "Classroom deleted successfully" });
    } catch (error) {
      console.error("Error deleting classroom:", error);
      res.status(500).json({ message: "Failed to delete classroom" });
    }
  }
);

/**
 * @route PUT /api/teacher/classroom/:id/evaluation
 * @desc Save evaluation settings (level, timer) without activating
 */
teacherRoutes.put(
  "/classroom/:id/evaluation",
  verifyToken,
  async (req: AuthRequest, res) => {
    try {
      const user = req.user as any;
      if (user?.role !== "teacher") {
        return res.status(403).json({ message: "Access denied. Teachers only." });
      }

      const classroom = await Classroom.findById(req.params.id);
      if (!classroom) {
        return res.status(404).json({ message: "Classroom not found" });
      }
      if (String(classroom.teacher) !== String(user.id)) {
        return res.status(403).json({ message: "Not your classroom" });
      }

      const { level, proctorTimerMinutes } = req.body;

      if (level && !["characters", "words", "both"].includes(level)) {
        return res.status(400).json({ message: "Invalid level. Must be 'characters', 'words', or 'both'." });
      }
      if (proctorTimerMinutes !== undefined && (proctorTimerMinutes < 5 || proctorTimerMinutes > 60)) {
        return res.status(400).json({ message: "Proctor timer must be between 5 and 60 minutes." });
      }

      if (level) classroom.activeEvaluation.level = level;
      if (proctorTimerMinutes !== undefined) classroom.activeEvaluation.proctorTimerMinutes = proctorTimerMinutes;

      await classroom.save();
      res.json({ message: "Evaluation settings updated", activeEvaluation: classroom.activeEvaluation });
    } catch (error) {
      console.error("Error updating evaluation settings:", error);
      res.status(500).json({ message: "Failed to update evaluation settings" });
    }
  }
);

/**
 * @route POST /api/teacher/classroom/:id/evaluation/activate
 * @desc Activate or deactivate the evaluation for a classroom
 */
teacherRoutes.post(
  "/classroom/:id/evaluation/activate",
  verifyToken,
  async (req: AuthRequest, res) => {
    try {
      const user = req.user as any;
      if (user?.role !== "teacher") {
        return res.status(403).json({ message: "Access denied. Teachers only." });
      }

      const classroom = await Classroom.findById(req.params.id);
      if (!classroom) {
        return res.status(404).json({ message: "Classroom not found" });
      }
      if (String(classroom.teacher) !== String(user.id)) {
        return res.status(403).json({ message: "Not your classroom" });
      }

      const { activate } = req.body;

      if (activate) {
        classroom.activeEvaluation.isActive = true;
        classroom.activeEvaluation.activatedAt = new Date();
      } else {
        classroom.activeEvaluation.isActive = false;
        classroom.activeEvaluation.activatedAt = null;
      }

      await classroom.save();
      res.json({
        message: activate ? "Evaluation activated" : "Evaluation deactivated",
        activeEvaluation: classroom.activeEvaluation,
      });
    } catch (error) {
      console.error("Error toggling evaluation:", error);
      res.status(500).json({ message: "Failed to toggle evaluation" });
    }
  }
);

/**
 * @route GET /api/teacher/classroom/:id/student/:studentId/results
 * @desc Get all typing results for a specific student in the classroom
 */
teacherRoutes.get(
  "/classroom/:id/student/:studentId/results",
  verifyToken,
  async (req: AuthRequest, res) => {
    try {
      const user = req.user as any;
      if (user?.role !== "teacher") {
        return res.status(403).json({ message: "Access denied. Teachers only." });
      }

      const classroom = await Classroom.findById(req.params.id);
      if (!classroom) {
        return res.status(404).json({ message: "Classroom not found" });
      }
      if (String(classroom.teacher) !== String(user.id)) {
        return res.status(403).json({ message: "Not your classroom" });
      }

      const { studentId } = req.params;
      const isEnrolled = classroom.students.some(
        (s: any) => String(s) === String(studentId)
      );
      if (!isEnrolled) {
        return res.status(404).json({ message: "Student not found in this classroom" });
      }

      // Result.userId stores the user's name (string), not ObjectId
      const student = await User.findById(studentId);
      if (!student) {
        return res.status(404).json({ message: "Student user not found" });
      }

      const results = await Result.find({ userId: student.name }).sort({ createdAt: 1 });
      res.json(results);
    } catch (error) {
      console.error("Error fetching student results:", error);
      res.status(500).json({ message: "Failed to fetch student results" });
    }
  }
);

export default teacherRoutes;