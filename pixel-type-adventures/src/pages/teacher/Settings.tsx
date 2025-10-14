import { useNavigate } from "react-router-dom";
import { Logo } from "@/components/Logo";
import { PixelButton } from "@/components/PixelButton";
import { PixelCard } from "@/components/PixelCard";
import { ArrowLeft, LogOut } from "lucide-react";
import { toast } from "sonner";

export default function TeacherSettings() {
  const navigate = useNavigate();
  const userName = localStorage.getItem("userName") || "Teacher";

  const handleLogout = () => {
    localStorage.clear();
    toast.success("Logged out successfully!");
    navigate("/");
  };

  return (
    <div className="min-h-screen p-8">
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
  );
}
