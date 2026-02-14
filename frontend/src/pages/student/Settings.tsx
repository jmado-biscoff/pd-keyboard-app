import { useNavigate } from "react-router-dom";
import { Logo } from "@/components/Logo";
import { PixelButton } from "@/components/PixelButton";
import { PixelCard } from "@/components/PixelCard";
import { PixelInput } from "@/components/PixelInput";
import { ArrowLeft, LogOut, Users, Volume2, Info } from "lucide-react";
import { toast } from "sonner";
import bgGif from "@/assets/b13.gif";
import { useState } from "react";
import {
  resolveProfileImage,
  getProfilesForRole,
} from "@/utils/profileAssets";
import fallbackProfilePic from "@/assets/cat-profile.jpg";
import { useAudio } from "@/contexts/AudioContext";

// Import about_us assets
import schoolLogo from "@/assets/about_us/school_logo .jpeg";
import pornobiPhoto from "@/assets/about_us/pornobi.png";
import cruzPhoto from "@/assets/about_us/cruz.png";
import deomampoPhoto from "@/assets/about_us/deomampo.png";
import gonzalesPhoto from "@/assets/about_us/gonzales.png";
import guraPhoto from "@/assets/about_us/gura.png";
import quejanoPhoto from "@/assets/about_us/quejano.jpeg";

export default function Settings() {
  const navigate = useNavigate();
  const [classroomCode, setClassroomCode] = useState("");
  const userName = localStorage.getItem("userName") || "Student";
  const [profileKey, setProfileKey] = useState(
    localStorage.getItem("profilePicture") || ""
  );
  const [showAvatarModal, setShowAvatarModal] = useState(false);
  const [tempSelectedProfile, setTempSelectedProfile] = useState(profileKey);
  const [showAboutUs, setShowAboutUs] = useState(false);

  // Audio
  const { musicVolume, soundVolume, setMusicVolume, setSoundVolume } = useAudio();

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

          {/* Audio Settings Section */}
          <PixelCard variant="accent" className="text-white">
            <h2 className="font-pixel text-xl mb-4 flex items-center gap-2">
              <Volume2 size={24} />
              Audio Settings
            </h2>

            {/* Background Music Volume */}
            <div className="mb-6">
              <label className="font-pixel text-sm mb-2 block">
                Background Music: {Math.round(musicVolume * 100)}%
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={musicVolume}
                onChange={(e) => setMusicVolume(parseFloat(e.target.value))}
                className="w-full h-2 bg-white/30 rounded-lg appearance-none cursor-pointer slider"
              />
            </div>

            {/* Sound Effects Volume */}
            <div>
              <label className="font-pixel text-sm mb-2 block">
                Sound Effects: {Math.round(soundVolume * 100)}%
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={soundVolume}
                onChange={(e) => setSoundVolume(parseFloat(e.target.value))}
                className="w-full h-2 bg-white/30 rounded-lg appearance-none cursor-pointer slider"
              />
            </div>
          </PixelCard>

          {/* About Us Section */}
          <PixelCard variant="yellow" className="text-black">
            <h2 className="font-pixel text-xl mb-4 flex items-center gap-2">
              <Info size={24} />
              About
            </h2>
            <p className="font-pixel text-xs mb-4 opacity-80">
              Learn more about the team behind TyPaw
            </p>
            <PixelButton
              variant="primary"
              onClick={() => setShowAboutUs(true)}
              className="w-full"
            >
              About Us
            </PixelButton>
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

      {/* About Us Modal */}
      {showAboutUs && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 overflow-y-auto">
          <PixelCard className="max-w-4xl w-full my-8">
            <div className="p-6 max-h-[85vh] overflow-y-auto">

              {/* School Logo & Title */}
              <div className="flex flex-col items-center mb-8">
                <img
                  src={schoolLogo}
                  alt="TIP-QC Logo"
                  className="w-32 h-32 object-contain mb-4"
                />
                <h2 className="font-pixel text-xl text-center mb-2">
                  Technological Institute of the Philippines - Quezon City
                </h2>
                <p className="font-pixel text-sm text-muted-foreground">
                  Project Design
                </p>
              </div>

              {/* Divider */}
              <div className="border-t-2 border-border/50 mb-6"></div>

              {/* Adviser Section */}
              <div className="mb-8">
                <h3 className="font-pixel text-lg text-center mb-6 text-purple-600">
                  Project Adviser
                </h3>
                <div className="flex flex-col items-center">
                  <img
                    src={pornobiPhoto}
                    alt="Engr. Lloyd Aldrin T. Pornobi"
                    className="w-40 h-40 rounded-lg object-cover border-4 border-purple-500/30 mb-3 shadow-lg"
                  />
                  <p className="font-pixel text-base font-bold text-foreground">
                    Engr. Lloyd Aldrin T. Pornobi
                  </p>
                  <p className="font-pixel text-xs text-muted-foreground italic">
                    Adviser
                  </p>
                </div>
              </div>

              {/* Divider */}
              <div className="border-t-2 border-border/50 mb-6"></div>

              {/* Team Members Section */}
              <div className="mb-8">
                <h3 className="font-pixel text-lg text-center mb-6 text-blue-600">
                  Development Team
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-6">

                  {/* Cruz */}
                  <div className="flex flex-col items-center text-center">
                    <img
                      src={cruzPhoto}
                      alt="Prince Ronbert Ledif C. Cruz"
                      className="w-28 h-28 rounded-lg object-cover border-2 border-blue-500/30 mb-2 shadow-md"
                    />
                    <p className="font-pixel text-xs font-semibold text-foreground leading-tight mb-1">
                      Cruz, Prince Ronbert Ledif C.
                    </p>
                    <p className="font-pixel text-[9px] text-muted-foreground italic">
                      Human-Computer Interaction
                    </p>
                  </div>

                  {/* De Omampo */}
                  <div className="flex flex-col items-center text-center">
                    <img
                      src={deomampoPhoto}
                      alt="Julius Mark A. De Omampo"
                      className="w-28 h-28 rounded-lg object-cover border-2 border-blue-500/30 mb-2 shadow-md"
                    />
                    <p className="font-pixel text-xs font-semibold text-foreground leading-tight mb-1">
                      De Omampo, Julius Mark A.
                    </p>
                    <p className="font-pixel text-[9px] text-muted-foreground italic">
                      System Administration
                    </p>
                  </div>

                  {/* Gonzales */}
                  <div className="flex flex-col items-center text-center">
                    <img
                      src={gonzalesPhoto}
                      alt="Allen Jerome Gonzales"
                      className="w-28 h-28 rounded-lg object-cover border-2 border-blue-500/30 mb-2 shadow-md"
                    />
                    <p className="font-pixel text-xs font-semibold text-foreground leading-tight mb-1">
                      Gonzales, Allen Jerome
                    </p>
                    <p className="font-pixel text-[9px] text-muted-foreground italic">
                      System Administration
                    </p>
                  </div>

                  {/* Gura */}
                  <div className="flex flex-col items-center text-center">
                    <img
                      src={guraPhoto}
                      alt="Cristine Canales Gura"
                      className="w-28 h-28 rounded-lg object-cover border-2 border-blue-500/30 mb-2 shadow-md"
                    />
                    <p className="font-pixel text-xs font-semibold text-foreground leading-tight mb-1">
                      Gura, Cristine Canales
                    </p>
                    <p className="font-pixel text-[9px] text-muted-foreground italic">
                      System Administration
                    </p>
                  </div>

                  {/* Quejano */}
                  <div className="flex flex-col items-center text-center">
                    <img
                      src={quejanoPhoto}
                      alt="Marc Angelo A. Quejano"
                      className="w-28 h-28 rounded-lg object-cover border-2 border-blue-500/30 mb-2 shadow-md"
                    />
                    <p className="font-pixel text-xs font-semibold text-foreground leading-tight mb-1">
                      Quejano, Marc Angelo A.
                    </p>
                    <p className="font-pixel text-[9px] text-muted-foreground italic">
                      Data Science
                    </p>
                  </div>
                </div>
              </div>

              {/* Divider */}
              <div className="border-t-2 border-border/50 mb-6"></div>

              {/* Special Thanks Section */}
              <div className="mb-6">
                <h3 className="font-pixel text-lg text-center mb-4 text-green-600">
                  Special Thanks
                </h3>

                {/* Client */}
                <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4 mb-3">
                  <p className="font-pixel text-xs text-green-600 font-bold mb-2">
                    Our Client:
                  </p>
                  <p className="font-pixel text-xs text-foreground">
                    Mr. Ryan Ricafort
                  </p>
                  <p className="font-pixel text-xs text-foreground">
                    Sto. Tomas de Villanueva Parochial School
                  </p>
                </div>

                {/* Panelists */}
                <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                  <p className="font-pixel text-xs text-purple-600 font-bold mb-2">
                    Our Panelists:
                  </p>
                  <p className="font-pixel text-xs text-foreground mb-1">
                    <span className="font-semibold">Lead Panel:</span> Dr. Roman M. Richard
                  </p>
                  <p className="font-pixel text-xs text-foreground">
                    <span className="font-semibold">Panel Members:</span>
                  </p>
                  <ul className="font-pixel text-xs text-foreground ml-4 mt-1 space-y-0.5">
                    <li>• Engr. Neal Barton James Matira</li>
                    <li>• Engr. Robin Valenzuela</li>
                  </ul>
                </div>
              </div>

              {/* Close Button */}
              <div className="flex justify-center mt-6">
                <PixelButton
                  variant="accent"
                  onClick={() => setShowAboutUs(false)}
                  className="px-8"
                >
                  Close
                </PixelButton>
              </div>
            </div>
          </PixelCard>
        </div>
      )}
    </div>
  );
}
