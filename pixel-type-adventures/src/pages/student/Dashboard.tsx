import { useNavigate } from "react-router-dom";
import { Logo } from "@/components/Logo";
import { PixelCard } from "@/components/PixelCard";
import learnIcon from "@/assets/file-logo.png";
import settingsIcon from "@/assets/settings-logo.png";
import playIcon from "@/assets/keyboard-logo.png";
import profilePic from "@/assets/cat-profile.jpg";

export default function StudentDashboard() {
  const navigate = useNavigate();
  const userName = localStorage.getItem("userName") || "Student";

  const menuItems = [
    {
      title: "Learn",
      icon: learnIcon,
      path: "/student/learn",
      variant: "yellow" as const,
      description: "Master touch typing",
    },
    {
      title: "Play",
      icon: playIcon,
      path: "/student/play",
      variant: "orange" as const,
      description: "Practice & compete",
    },
    {
      title: "Settings",
      icon: settingsIcon,
      path: "/student/settings",
      variant: "purple" as const,
      description: "Manage account",
    },
  ];

  return (
    <div className="min-h-screen p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex justify-between items-center mb-12">
          <Logo />
          <div className="flex items-center gap-3">
            {/* Profile Picture */}
            <img
              src={profilePic}
              alt="Profile"
              className="h-12 w-12 rounded-md border-2 border-black object-cover image-render-pixel"
            />
            <div>
              <p className="font-pixel text-xs text-muted-foreground">
                Student
              </p>
              <p className="font-pixel text-sm">{userName}</p>
            </div>
          </div>
        </div>

        {/* Menu Grid */}
        <div className="grid md:grid-cols-3 gap-10 justify-center place-items-center">
          {menuItems.map((item) => (
            <button
              key={item.path}
              onClick={() => navigate(item.path)}
              className="transition-transform hover:scale-105 active:scale-95"
            >
              <PixelCard
                variant={item.variant}
                className="w-64 h-56 flex flex-col items-center justify-center gap-4 p-8 text-white"
              >
                <img
                  src={item.icon}
                  alt={item.title}
                  className="h-12 w-auto select-none image-render-pixel"
                />
                <h2 className="font-pixel text-xl">{item.title}</h2>
                <p className="font-pixel text-xs opacity-90 text-center">
                  {item.description}
                </p>
              </PixelCard>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
