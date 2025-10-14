import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { PixelButton } from "@/components/PixelButton";
import { PixelCard } from "@/components/PixelCard";
import { PixelInput } from "@/components/PixelInput";
import { Logo } from "@/components/Logo";
import { toast } from "sonner";
import bgGif from "@/assets/bg.gif";

type UserType = "student" | "teacher";
type AuthMode = "login" | "signup";

export default function Auth() {
  const navigate = useNavigate();
  const [authMode, setAuthMode] = useState<AuthMode>("login");
  const [userType, setUserType] = useState<UserType>("student");
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
    confirmPassword: "",
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (authMode === "signup") {
      if (formData.password !== formData.confirmPassword) {
        toast.error("Passwords don't match!");
        return;
      }
      if (!formData.name || !formData.email || !formData.password) {
        toast.error("Please fill all fields!");
        return;
      }
    }

    // Store user type in localStorage (temporary, will use backend later)
    localStorage.setItem("userType", userType);
    localStorage.setItem("userName", formData.name || formData.email);
    
    toast.success(`Welcome ${userType === "student" ? "Student" : "Teacher"}!`);
    
    // Navigate to appropriate dashboard
    if (userType === "student") {
      navigate("/student/dashboard");
    } else {
      navigate("/teacher/dashboard");
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-no-repeat bg-center bg-cover" style={{ backgroundImage: `url(${bgGif})` }}>
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
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  placeholder="Your Name"
                  required={authMode === "signup"}
                />
              </div>
            )}

            <div>
              <label className="block font-pixel text-xs mb-2">Email</label>
              <PixelInput
                type="email"
                value={formData.email}
                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                placeholder="email@example.com"
                required
              />
            </div>

            <div>
              <label className="block font-pixel text-xs mb-2">Password</label>
              <PixelInput
                type="password"
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                placeholder="••••••••"
                required
              />
            </div>

            {authMode === "signup" && (
              <div>
                <label className="block font-pixel text-xs mb-2">Confirm Password</label>
                <PixelInput
                  type="password"
                  value={formData.confirmPassword}
                  onChange={(e) => setFormData({ ...formData, confirmPassword: e.target.value })}
                  placeholder="••••••••"
                  required={authMode === "signup"}
                />
              </div>
            )}

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

            <PixelButton type="submit" variant="accent" className="w-full">
              {authMode === "login" ? "Login" : "Sign Up"}
            </PixelButton>
          </form>

          <div className="mt-4 text-center">
            <button
              type="button"
              onClick={() => setAuthMode(authMode === "login" ? "signup" : "login")}
              className="font-pixel text-xs text-white hover:text-accent transition-colors underline"
            >
              {authMode === "login" ? "Don't have an account? Sign Up" : "Already have an account? Login"}
            </button>
          </div>
        </PixelCard>
      </div>
    </div>
  );
}
