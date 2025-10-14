import { useNavigate } from "react-router-dom";
import { Logo } from "@/components/Logo";
import { PixelButton } from "@/components/PixelButton";
import { PixelCard } from "@/components/PixelCard";
import { PixelInput } from "@/components/PixelInput";
import { ArrowLeft, LogOut, Users } from "lucide-react";
import { toast } from "sonner";
import { useState } from "react";

export default function Settings() {
  const navigate = useNavigate();
  const [classroomCode, setClassroomCode] = useState("");
  const userName = localStorage.getItem("userName") || "Student";

  const handleLogout = () => {
    localStorage.clear();
    toast.success("Logged out successfully!");
    navigate("/");
  };

  const handleJoinClassroom = () => {
    if (classroomCode.trim()) {
      toast.success(`Joined classroom: ${classroomCode}`);
      setClassroomCode("");
    } else {
      toast.error("Please enter a classroom code");
    }
  };

  return (
    <div className="min-h-screen p-8">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center gap-4 mb-12">
          <PixelButton variant="secondary" onClick={() => navigate("/student/dashboard")}>
            <ArrowLeft size={20} />
          </PixelButton>
          <Logo />
        </div>

        <div className="space-y-6">
          {/* Profile */}
          <PixelCard>
            <h2 className="font-pixel text-xl mb-4">Profile</h2>
            <div className="flex items-center gap-4">
              <div className="text-6xl">🦁</div>
              <div>
                <p className="font-pixel text-sm text-muted-foreground">Student Name</p>
                <p className="font-pixel text-lg">{userName}</p>
              </div>
            </div>
          </PixelCard>

          {/* Classroom */}
          <PixelCard variant="purple" className="text-white">
            <h2 className="font-pixel text-xl mb-4 flex items-center gap-2">
              <Users size={24} />
              Join Classroom
            </h2>
            <p className="font-pixel text-xs mb-4 opacity-90">
              Enter the code provided by your teacher to join their classroom
            </p>
            <div className="flex gap-2">
              <PixelInput
                value={classroomCode}
                onChange={(e) => setClassroomCode(e.target.value.toUpperCase())}
                placeholder="CLASSROOM CODE"
                className="flex-1"
                maxLength={8}
              />
              <PixelButton variant="accent" onClick={handleJoinClassroom}>
                Join
              </PixelButton>
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
