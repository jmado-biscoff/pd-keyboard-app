import { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { PixelButton } from "@/components/PixelButton";
import { PixelCard } from "@/components/PixelCard";
import { PixelInput } from "@/components/PixelInput";
import { Logo } from "@/components/Logo";
import { toast } from "sonner";
import { Camera, Monitor, CheckCircle, Eye } from "lucide-react";
import bgGif from "@/assets/bg.gif";
import { getRandomProfile } from "@/utils/profileAssets";

type UserType = "student" | "teacher";
type AuthMode = "login" | "signup";

export default function Auth() {
  const navigate = useNavigate();
  const [authMode, setAuthMode] = useState<AuthMode>("login");
  const [userType, setUserType] = useState<UserType>("student");
  const [loading, setLoading] = useState(false);
  const [showSetupGuide, setShowSetupGuide] = useState(false);
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

        {/* ✅ NEW: Setup Guide Button - Bottom Right */}
        <div className="fixed bottom-6 right-6 z-10">
          <PixelButton
            variant="secondary"
            size="sm"
            onClick={() => setShowSetupGuide(true)}
            className="flex items-center gap-2 shadow-lg"
          >
            <Camera size={16} />
            Setup Guide
          </PixelButton>
        </div>
      </div>

      {/* ✅ NEW: Setup Guide Modal */}
      {showSetupGuide && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
          <PixelCard className="max-w-3xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <h2 className="font-pixel text-2xl mb-6 text-center">
                Camera Setup Guide
              </h2>

              {/* 4-Panel Grid */}
              <div className="grid md:grid-cols-2 gap-4 mb-6">
                {/* Panel 1: Top-View Position */}
                <div className="bg-blue-500/10 border-2 border-blue-400/50 rounded-lg p-4">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="bg-blue-500 rounded-full p-2">
                      <Camera size={24} className="text-white" />
                    </div>
                    <h3 className="font-pixel text-sm text-blue-400">
                      1. Top-View Position
                    </h3>
                  </div>
                  <p className="font-pixel text-xs text-muted-foreground leading-relaxed">
                    Position your camera directly above your keyboard in a bird's-eye view.
                    The camera should capture the entire keyboard from top to bottom.
                  </p>
                </div>

                {/* Panel 2: Keep Keyboard Still */}
                <div className="bg-green-500/10 border-2 border-green-400/50 rounded-lg p-4">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="bg-green-500 rounded-full p-2">
                      <Monitor size={24} className="text-white" />
                    </div>
                    <h3 className="font-pixel text-sm text-green-400">
                      2. Keyboard Stays Still
                    </h3>
                  </div>
                  <p className="font-pixel text-xs text-muted-foreground leading-relaxed">
                    Once positioned, do NOT move your keyboard or camera during the session.
                    Movement will break calibration and require restarting.
                  </p>
                </div>

                {/* Panel 3: Camera Permission */}
                <div className="bg-purple-500/10 border-2 border-purple-400/50 rounded-lg p-4">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="bg-purple-500 rounded-full p-2">
                      <CheckCircle size={24} className="text-white" />
                    </div>
                    <h3 className="font-pixel text-sm text-purple-400">
                      3. Allow Camera Permission
                    </h3>
                  </div>
                  <p className="font-pixel text-xs text-muted-foreground leading-relaxed">
                    Your browser will ask for camera access. Click "Allow" to enable
                    hand-tracking detection. We never record or store video.
                  </p>
                </div>

                {/* Panel 4: Mirror View */}
                <div className="bg-orange-500/10 border-2 border-orange-400/50 rounded-lg p-4">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="bg-orange-500 rounded-full p-2">
                      <Eye size={24} className="text-white" />
                    </div>
                    <h3 className="font-pixel text-sm text-orange-400">
                      4. Mirror View Check
                    </h3>
                  </div>
                  <p className="font-pixel text-xs text-muted-foreground leading-relaxed">
                    Ensure the camera view mirrors your keyboard exactly as it appears
                    in front of you. All keys should be clearly visible.
                  </p>
                </div>
              </div>

              {/* Additional Tips */}
              <div className="bg-yellow-500/10 border-2 border-yellow-400/50 rounded-lg p-4 mb-6">
                <p className="font-pixel text-xs text-yellow-400 mb-2">💡 Pro Tips:</p>
                <ul className="font-pixel text-[10px] text-muted-foreground space-y-1 leading-relaxed">
                  <li>• Use good lighting - avoid shadows on the keyboard</li>
                  <li>• Remove hands during initialization phase</li>
                  <li>• Use a stable camera mount or tripod for best results</li>
                  <li>• Test your setup before starting a graded session</li>
                </ul>
              </div>

              {/* Close Button */}
              <div className="flex justify-center">
                <PixelButton
                  variant="accent"
                  onClick={() => setShowSetupGuide(false)}
                  className="px-8"
                >
                  Got it!
                </PixelButton>
              </div>
            </div>
          </PixelCard>
        </div>
      )}
    </div>
  );
}
