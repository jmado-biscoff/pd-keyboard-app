//comment
import express, { Request, Response } from "express";
import jwt from "jsonwebtoken";
import bcrypt from "bcryptjs";
import { User } from "../models/User";
import { verifyToken, AuthRequest } from "../../middleware/authMiddleware";

const authRoutes = express.Router();

/**
 * @route POST /api/auth/register
 * @desc Register a new user
 */
authRoutes.post(
  "/register",
  async (req: Request, res: Response): Promise<void> => {
    try {
      const { name, email, password, role, profilePicture } = req.body;

      if (!name || !email || !password || !role) {
        res.status(400).json({ message: "All fields are required." });
        return;
      }

      const existingUser = await User.findOne({ email });
      if (existingUser) {
        res.status(400).json({ message: "Email already exists." });
        return;
      }

      const hashedPassword = await bcrypt.hash(password, 10);
      const user = new User({
        name,
        email,
        password: hashedPassword,
        role,
        profilePicture: profilePicture || "",
      });
      await user.save();

      const token = jwt.sign(
        { id: user._id, email: user.email, role: user.role },
        process.env.JWT_SECRET!,
        { expiresIn: "7d" }
      );

      res.status(201).json({
        message: "Account created successfully!",
        token,
        user: {
          id: user._id,
          name: user.name,
          email: user.email,
          role: user.role,
          profilePicture: user.profilePicture,
        },
      });
    } catch (error) {
      console.error("Signup Error:", error);
      res.status(500).json({ message: "Server error during signup.", error });
    }
  }
);

/**
 * @route POST /api/auth/login
 * @desc Authenticate user
 */
authRoutes.post(
  "/login",
  async (req: Request, res: Response): Promise<void> => {
    try {
      const { email, password } = req.body;

      if (!email || !password) {
        res.status(400).json({ message: "Email and password are required." });
        return;
      }

      const user = await User.findOne({ email });
      if (!user) {
        res.status(400).json({ message: "User not found." });
        return;
      }

      const isMatch = await bcrypt.compare(password, user.password);
      if (!isMatch) {
        res.status(400).json({ message: "Invalid email or password." });
        return;
      }

      const token = jwt.sign(
        { id: user._id, email: user.email, role: user.role },
        process.env.JWT_SECRET!,
        { expiresIn: "7d" }
      );

      res.status(200).json({
        message: "Login successful!",
        token,
        user: {
          id: user._id,
          name: user.name,
          email: user.email,
          role: user.role,
          profilePicture: user.profilePicture,
        },
      });
    } catch (error) {
      console.error("Login Error:", error);
      res.status(500).json({ message: "Server error during login.", error });
    }
  }
);

/**
 * @route PUT /api/auth/update-profile
 * @desc Update user profile picture
 */
authRoutes.put(
  "/update-profile",
  verifyToken,
  async (req: AuthRequest, res: Response): Promise<void> => {
    try {
      const { profilePicture } = req.body;
      const userId = (req.user as jwt.JwtPayload)?.id;

      if (!userId) {
        res.status(401).json({ message: "Unauthorized." });
        return;
      }

      const user = await User.findByIdAndUpdate(
        userId,
        { profilePicture },
        { new: true }
      );

      if (!user) {
        res.status(404).json({ message: "User not found." });
        return;
      }

      res.status(200).json({
        message: "Profile updated successfully!",
        user: {
          id: user._id,
          name: user.name,
          email: user.email,
          role: user.role,
          profilePicture: user.profilePicture,
        },
      });
    } catch (error) {
      console.error("Profile Update Error:", error);
      res.status(500).json({ message: "Server error updating profile.", error });
    }
  }
);

export default authRoutes;
