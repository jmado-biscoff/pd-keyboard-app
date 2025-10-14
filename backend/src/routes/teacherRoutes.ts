import express from "express";
import { verifyToken, AuthRequest } from "../../middleware/authMiddleware";

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

export default teacherRoutes;
