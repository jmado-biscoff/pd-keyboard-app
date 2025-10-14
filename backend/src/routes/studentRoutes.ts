import express from "express";
import { verifyToken, AuthRequest } from "../../middleware/authMiddleware";

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

export default studentRoutes;
