import { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { PixelButton } from "@/components/PixelButton";
import { PixelCard } from "@/components/PixelCard";
import { PixelInput } from "@/components/PixelInput";
import { Logo } from "@/components/Logo";
import { toast } from "sonner";
import bgGif from "@/assets/bg.gif";
import { getRandomProfile } from "@/utils/profileAssets";

type UserType = "student" | "teacher";
type AuthMode = "login" | "signup";

export default function Auth() {
  const navigate = useNavigate();
  const [authMode, setAuthMode] = useState<AuthMode>("login");
  const [userType, setUserType] = useState<UserType>("student");
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
    confirmPassword: "",
  });

  // ✅ Use the same URL as backend
  const API_URL =
    import.meta.env.VITE_API_URL || "http://localhost:5000/api/auth";

  const handleChange = (key: string, value: string) => {
    setFormData((prev) => ({ ...prev, [key]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // basic validation
    if (!formData.email || !formData.password) {
      toast.error("Please fill all required fields!");
      return;
    }

    if (authMode === "signup") {
      if (!formData.name || !formData.confirmPassword) {
        toast.error("Please complete all sign-up fields!");
        return;
      }
      if (formData.password !== formData.confirmPassword) {
        toast.error("Passwords do not match!");
        return;
      }
    }

    setLoading(true);

    try {
      const endpoint =
        authMode === "signup" ? `${API_URL}/register` : `${API_URL}/login`;

      const payload =
        authMode === "signup"
          ? {
              name: formData.name,
              email: formData.email,
              password: formData.password,
              role: userType,
              profilePicture: getRandomProfile(userType),
            }
          : {
              email: formData.email,
              password: formData.password,
            };

      // ✅ send POST request
      const res = await axios.post(endpoint, payload, {
        headers: { "Content-Type": "application/json" },
      });

      const data = res.data;

      toast.success(
        authMode === "signup"
          ? "Account created successfully!"
          : "Login successful!"
      );

      // save token and user info
      if (data?.token) localStorage.setItem("token", data.token);
      if (data?.user) {
        localStorage.setItem("userName", data.user.name || formData.name);
        localStorage.setItem("userType", data.user.role || userType);
        localStorage.setItem(
          "profilePicture",
          data.user.profilePicture || ""
        );
      }

      const role = data?.user?.role || userType;
      navigate(
        role === "teacher" ? "/teacher/dashboard" : "/student/dashboard"
      );
    } catch (err: any) {
      console.error("❌ Auth error:", err);

      // network fallback
      if (err.code === "ERR_NETWORK" || err.message?.includes("Network")) {
        toast.warning("⚠️ Backend not reachable — using local login mode.");
        localStorage.setItem("userType", userType);
        localStorage.setItem("userName", formData.name || formData.email);
        navigate(
          userType === "student" ? "/student/dashboard" : "/teacher/dashboard"
        );
        return;
      }

      const msg =
        err.response?.data?.message ||
        (authMode === "signup"
          ? "Failed to create account."
          : "Invalid credentials.");
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className="min-h-screen flex items-center justify-center p-4 bg-no-repeat bg-center bg-cover"
      style={{ backgroundImage: `url(${bgGif})` }}
    >
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <Logo className="justify-center mb-4" />
        </div>

        <PixelCard variant="orange" className="text-white">
          <h2 className="font-pixel text-xl mb-6 text-center">
            {authMode === "login" ? "Login" : "Create Account"}
          </h2>

          <form onSubmit={handleSubmit} className="space-y-4">
            {authMode === "signup" && (
              <div>
                <label className="block font-pixel text-xs mb-2">Name</label>
                <PixelInput
                  type="text"
                  value={formData.name}
                  onChange={(e) => handleChange("name", e.target.value)}
                  placeholder="Your Name"
                  required
                />
              </div>
            )}

            <div>
              <label className="block font-pixel text-xs mb-2">Email</label>
              <PixelInput
                type="email"
                value={formData.email}
                onChange={(e) => handleChange("email", e.target.value)}
                placeholder="email@example.com"
                required
              />
            </div>

            <div>
              <label className="block font-pixel text-xs mb-2">Password</label>
              <PixelInput
                type="password"
                value={formData.password}
                onChange={(e) => handleChange("password", e.target.value)}
                placeholder="••••••••"
                required
              />
            </div>

            {authMode === "signup" && (
              <div>
                <label className="block font-pixel text-xs mb-2">
                  Confirm Password
                </label>
                <PixelInput
                  type="password"
                  value={formData.confirmPassword}
                  onChange={(e) =>
                    handleChange("confirmPassword", e.target.value)
                  }
                  placeholder="••••••••"
                  required
                />
              </div>
            )}

            {/* Role selector — only visible during signup */}
            {authMode === "signup" && (
              <div>
                <label className="block font-pixel text-xs mb-2">I am a</label>
                <div className="flex gap-2">
                  <PixelButton
                    type="button"
                    variant={userType === "student" ? "accent" : "primary"}
                    className="flex-1"
                    onClick={() => setUserType("student")}
                  >
                    Student
                  </PixelButton>
                  <PixelButton
                    type="button"
                    variant={userType === "teacher" ? "accent" : "primary"}
                    className="flex-1"
                    onClick={() => setUserType("teacher")}
                  >
                    Teacher
                  </PixelButton>
                </div>
              </div>
            )}

            <PixelButton
              type="submit"
              variant="accent"
              className="w-full"
              disabled={loading}
            >
              {loading
                ? "Please wait..."
                : authMode === "login"
                ? "Login"
                : "Sign Up"}
            </PixelButton>
          </form>

          <div className="mt-4 text-center">
            <button
              type="button"
              onClick={() =>
                setAuthMode(authMode === "login" ? "signup" : "login")
              }
              className="font-pixel text-xs text-white hover:text-accent transition-colors underline"
            >
              {authMode === "login"
                ? "Don't have an account? Sign Up"
                : "Already have an account? Login"}
            </button>
          </div>
        </PixelCard>
      </div>
    </div>
  );
}
