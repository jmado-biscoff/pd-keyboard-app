import { useNavigate } from "react-router-dom";
import { Logo } from "@/components/Logo";
import { PixelButton } from "@/components/PixelButton";
import { PixelCard } from "@/components/PixelCard";
import { ArrowLeft, LogOut } from "lucide-react";
import { toast } from "sonner";
import bgVideo from "@/assets/b12.mp4";

export default function TeacherSettings() {
  const navigate = useNavigate();
  const userName = localStorage.getItem("userName") || "Teacher";

  const handleLogout = () => {
    localStorage.clear();
    toast.success("Logged out successfully!");
    navigate("/");
  };

  return (
    <div className="relative min-h-screen overflow-hidden">
      {/* Background Video */}
      <video
        autoPlay
        loop
        muted
        playsInline
        className="absolute top-0 left-0 w-full h-full object-cover -z-10"
      >
        <source src={bgVideo} type="video/mp4" />
      </video>

      {/* Page Content */}
      <div className="relative z-10 p-8 bg-black/20 min-h-screen">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center gap-4 mb-12">
            <PixelButton variant="secondary" onClick={() => navigate("/teacher/dashboard")}>
              <ArrowLeft size={20} />
            </PixelButton>
            <Logo />
          </div>

          <div className="space-y-6">
            {/* Profile */}
            <PixelCard>
              <h2 className="font-pixel text-xl mb-4">Profile</h2>
              <div className="flex items-center gap-4">
                <div className="text-6xl">👨‍🏫</div>
                <div>
                  <p className="font-pixel text-sm text-muted-foreground">Teacher Name</p>
                  <p className="font-pixel text-lg">{userName}</p>
                </div>
              </div>
            </PixelCard>

            {/* Logout */}
            <PixelCard variant="orange" className="text-white">
              <h2 className="font-pixel text-xl mb-4">Account</h2>
              <PixelButton
                variant="accent"
                onClick={handleLogout}
                className="w-full flex items-center justify-center gap-2"
              >
                <LogOut size={20} />
                Logout
              </PixelButton>
            </PixelCard>
          </div>
        </div>
      </div>
    </div>
  );
}
