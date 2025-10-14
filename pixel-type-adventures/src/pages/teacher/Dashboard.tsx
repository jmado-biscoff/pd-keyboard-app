import { useNavigate } from "react-router-dom";
import { Logo } from "@/components/Logo";
import { PixelCard } from "@/components/PixelCard";
import { Users, Settings } from "lucide-react";

export default function TeacherDashboard() {
  const navigate = useNavigate();
  const userName = localStorage.getItem("userName") || "Teacher";

  const menuItems = [
    {
      title: "Classroom",
      icon: Users,
      path: "/teacher/classroom",
      variant: "orange" as const,
      description: "Manage students",
    },
    {
      title: "Settings",
      icon: Settings,
      path: "/teacher/settings",
      variant: "purple" as const,
      description: "Account settings",
    },
  ];

  return (
    <div className="min-h-screen p-8">
      <div className="max-w-6xl mx-auto">
        <div className="flex justify-between items-center mb-12">
          <Logo />
          <div className="flex items-center gap-3">
            <div className="text-4xl">👨‍🏫</div>
            <div>
              <p className="font-pixel text-xs text-muted-foreground">Teacher</p>
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
                <item.icon size={80} strokeWidth={2.5} />
                <h2 className="font-pixel text-2xl">{item.title}</h2>
                <p className="font-pixel text-xs opacity-90">{item.description}</p>
              </PixelCard>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
