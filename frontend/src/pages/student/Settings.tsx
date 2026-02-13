import { useNavigate } from "react-router-dom";
import { Logo } from "@/components/Logo";
import { PixelButton } from "@/components/PixelButton";
import { PixelCard } from "@/components/PixelCard";
import { PixelInput } from "@/components/PixelInput";
import { ArrowLeft, LogOut, Users } from "lucide-react";
import { toast } from "sonner";
import bgGif from "@/assets/b13.gif";
import { useState } from "react";
import {
  resolveProfileImage,
  getProfilesForRole,
} from "@/utils/profileAssets";
import fallbackProfilePic from "@/assets/cat-profile.jpg";

export default function Settings() {
  const navigate = useNavigate();
  const [classroomCode, setClassroomCode] = useState("");
  const userName = localStorage.getItem("userName") || "Student";
  const [profileKey, setProfileKey] = useState(
    localStorage.getItem("profilePicture") || ""
  );
  const [showAvatarModal, setShowAvatarModal] = useState(false);
  const [tempSelectedProfile, setTempSelectedProfile] = useState(profileKey);

  const profilePic = resolveProfileImage(profileKey) || fallbackProfilePic;
  const availableProfiles = getProfilesForRole("student");

  const handleAvatarSelect = async (key: string) => {
    const token = localStorage.getItem("token");
    if (!token) {
      toast.error("Please log in first");
      return;
    }

    try {
      const API_URL =
        import.meta.env.VITE_API_URL || "http://localhost:5000/api/auth";
      const res = await fetch(`${API_URL}/update-profile`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ profilePicture: key }),
      });

      const data = await res.json();

      if (res.ok) {
        setProfileKey(key);
        localStorage.setItem("profilePicture", key);
        setShowAvatarModal(false);
        toast.success("Avatar updated!");
      } else {
        toast.error(data.message || "Failed to update avatar");
      }
    } catch {
      toast.error("Error updating avatar");
    }
  };

  const openAvatarModal = () => {
    setTempSelectedProfile(profileKey);
    setShowAvatarModal(true);
  };

  const closeAvatarModal = () => {
    setTempSelectedProfile(profileKey);
    setShowAvatarModal(false);
  };

  // Logout Functionality
  const handleLogout = () => {
    localStorage.clear();
    toast.success("Logged out successfully!");
    navigate("/");
  };

  // Join Classroom (connected to backend)
  const handleJoinClassroom = async () => {
    if (!classroomCode.trim()) {
      toast.error("Please enter a classroom code");
      return;
    }

    const token = localStorage.getItem("token");
    if (!token) {
      toast.error("Please log in first");
      return;
    }

    try {
      const res = await fetch(
        `${import.meta.env.VITE_API_URL}/student/join-classroom`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({ code: classroomCode }),
        }
      );

      const data = await res.json();

      if (res.ok) {
        toast.success(`Joined classroom: ${data.classroom.name}`);
        setClassroomCode("");
      } else {
        toast.error(data.message || "Failed to join classroom");
      }
    } catch (error) {
      console.error("Error joining classroom:", error);
      toast.error("Error joining classroom");
    }
  };

  return (
    <div className="relative min-h-screen overflow-hidden">
      {/* Background GIF */}
      <img
        src={bgGif}
        alt="Background"
        className="absolute top-0 left-0 w-full h-full object-cover -z-10"
      />

      <div className="relative z-10 p-8 bg-black/20 min-h-screen">
      <div className="max-w-4xl mx-auto">
        {/* Back Button + Logo */}
        <div className="flex items-center gap-4 mb-12">
          <PixelButton
            variant="secondary"
            onClick={() => navigate("/student/dashboard")}
          >
            <ArrowLeft size={20} />
          </PixelButton>
          <Logo />
        </div>

        <div className="space-y-6">
          {/* Profile Section */}
          <PixelCard>
            <h2 className="font-pixel text-xl mb-4">Profile</h2>
            <div className="flex items-center gap-4">
              <img
                src={profilePic}
                alt="Profile"
                className="h-16 w-16 rounded-md border-2 border-black object-cover image-render-pixel"
              />
              <div>
                <p className="font-pixel text-sm text-muted-foreground">
                  Student Name
                </p>
                <p className="font-pixel text-lg">{userName}</p>
              </div>
              <PixelButton
                variant="secondary"
                size="sm"
                onClick={openAvatarModal}
                className="ml-auto"
              >
                Edit Avatar
              </PixelButton>
            </div>
          </PixelCard>

          {/* Classroom Section */}
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

          {/* Logout Section */}
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

      {/* Avatar Selection Modal */}
      {showAvatarModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80">
          <PixelCard className="max-w-lg w-full mx-4">
            <h2 className="font-pixel text-xl mb-6 text-center">
              Choose Your Avatar
            </h2>
            <div className="grid grid-cols-3 sm:grid-cols-3 gap-4 mb-6">
              {availableProfiles.map((p) => (
                <button
                  key={p.key}
                  onClick={() => setTempSelectedProfile(p.key)}
                  className={`rounded-md border-3 overflow-hidden transition-transform hover:scale-105 ${tempSelectedProfile === p.key
                      ? "border-accent ring-2 ring-accent scale-105"
                      : "border-black"
                    }`}
                >
                  <img
                    src={p.src}
                    alt={p.key}
                    className="w-full aspect-square object-cover image-render-pixel"
                  />
                </button>
              ))}
            </div>
            <div className="flex gap-3 justify-end">
              <PixelButton variant="secondary" onClick={closeAvatarModal}>
                Cancel
              </PixelButton>
              <PixelButton
                variant="accent"
                onClick={() => handleAvatarSelect(tempSelectedProfile)}
              >
                Save Changes
              </PixelButton>
            </div>
          </PixelCard>
        </div>
      )}
    </div>
  );
}
