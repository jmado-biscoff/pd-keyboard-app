//comment
import { Request, Response, NextFunction } from "express";
import jwt, { JwtPayload } from "jsonwebtoken";

// Extend Express Request to include user data
export interface AuthRequest extends Request {
  user?: string | JwtPayload;
}

export const verifyToken = (
  req: AuthRequest,
  res: Response,
  next: NextFunction
): void => {
  try {
    const authHeader = req.headers.authorization;

    if (!authHeader || !authHeader.startsWith("Bearer ")) {
      res.status(401).json({ message: "Access denied. No token provided." });
      return;
    }

    const token = authHeader.split(" ")[1];
    const secret = process.env.JWT_SECRET;

    if (!secret) {
      console.error("❌ JWT_SECRET not set in environment variables.");
      res.status(500).json({ message: "Server configuration error." });
      return;
    }

    const decoded = jwt.verify(token, secret);
    req.user = decoded;

    next();
  } catch (error: any) {
    console.error("❌ Token verification error:", error.message);
    res.status(401).json({ message: "Invalid or expired token." });
  }
};
