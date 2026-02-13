import { useNavigate } from "react-router-dom";
import { Logo } from "@/components/Logo";
import { PixelCard } from "@/components/PixelCard";
import bgVideo from "@/assets/b10.mp4";
import classroomLogo from "@/assets/classroom.png";
import settingsLogo from "@/assets/settings-logo.png";
import { resolveProfileImage } from "@/utils/profileAssets";
import fallbackProfilePic from "@/assets/cat-profile.jpg";

export default function TeacherDashboard() {
  const navigate = useNavigate();
  const userName = localStorage.getItem("userName") || "Teacher";
  const profileKey = localStorage.getItem("profilePicture") || "";
  const profilePic = resolveProfileImage(profileKey) || fallbackProfilePic;

  const menuItems = [
    {
      title: "Classroom",
      logo: classroomLogo,
      path: "/teacher/classroom",
      variant: "orange" as const,
      description: "Manage students",
    },
    {
      title: "Settings",
      logo: settingsLogo,
      path: "/teacher/settings",
      variant: "purple" as const,
      description: "Account settings",
    },
  ];

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
        <div className="max-w-6xl mx-auto">
          <div className="flex justify-between items-center mb-12">
            <Logo />
            <div className="flex items-center gap-3">
              <img
                src={profilePic}
                alt="Profile"
                className="h-12 w-12 rounded-md border-2 border-black object-cover image-render-pixel"
              />
              <div>
                <p className="font-pixel text-xs text-black">Teacher</p>
                <p className="font-pixel text-sm">{userName}</p>
              </div>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
            {menuItems.map((item) => (
              <button
                key={item.path}
                onClick={() => navigate(item.path)}
                className="text-left transition-transform hover:scale-105 active:scale-95"
              >
                <PixelCard variant={item.variant} className="h-full flex flex-col items-center justify-center gap-4 p-12 text-white">
                  <img src={item.logo} alt={item.title} className="w-20 h-20" />
                  <h2 className="font-pixel text-2xl">{item.title}</h2>
                  <p className="font-pixel text-xs opacity-90">{item.description}</p>
                </PixelCard>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
