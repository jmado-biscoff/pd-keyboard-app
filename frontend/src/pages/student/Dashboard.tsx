import { useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Logo } from "@/components/Logo";
import { PixelCard } from "@/components/PixelCard";
import learnIcon from "@/assets/file-logo.png";
import settingsIcon from "@/assets/settings-logo.png";
import playIcon from "@/assets/keyboard-logo.png";
import profilePic from "@/assets/cat-profile.jpg";
import bgVideo from "@/assets/b4.mp4";

export default function StudentDashboard() {
  const navigate = useNavigate();
  const userName = localStorage.getItem("userName") || "Student";
  const [classrooms, setClassrooms] = useState<any[]>([]);

  // 🔹 Fetch classrooms the student has joined
  useEffect(() => {
    const fetchClassrooms = async () => {
      const token = localStorage.getItem("token");
      try {
        const res = await fetch(
          `${import.meta.env.VITE_API_URL}/student/my-classrooms`,
          {
            headers: { Authorization: `Bearer ${token}` },
          }
        );
        const data = await res.json();
        if (res.ok) setClassrooms(data);
        else toast.error(data.message || "Failed to load classrooms");
      } catch {
        toast.error("Error fetching classrooms");
      }
    };
    fetchClassrooms();
  }, []);

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
      variant: "red" as const,
      description: "Manage account",
    },
  ];

  return (
    <div className="relative min-h-screen overflow-hidden">
      {/* 🔹 Background Video */}
      <video
        autoPlay
        loop
        muted
        playsInline
        className="absolute top-0 left-0 w-full h-full object-cover -z-10"
      >
        <source src={bgVideo} type="video/mp4" />
      </video>

      {/* 🔹 Page Content */}
      <div className="relative z-10 p-8 bg-black/20 min-h-screen">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <div className="flex justify-between items-center mb-12">
            <div className="scale-125">
              <Logo />
            </div>
            <div className="flex items-center gap-3">
              <img
                src={profilePic}
                alt="Profile"
                className="h-12 w-12 rounded-md border-2 border-black object-cover image-render-pixel"
              />
              <div>
                <p className="font-pixel text-xs text-black">Student</p>
                <p className="font-pixel text-sm">{userName}</p>
              </div>
            </div>
          </div>

          {/* 🟣 My Classrooms Section */}
          <PixelCard variant="purple" className="text-white mb-10">
            <h2 className="font-pixel text-xl mb-4">My Classrooms</h2>
            {classrooms.length > 0 ? (
              <ul className="space-y-2">
                {classrooms.map((cls) => (
                  <li
                    key={cls._id}
                    className="font-pixel text-sm flex justify-between border-b border-white/20 pb-1"
                  >
                    <span>{cls.name}</span>
                    <span className="opacity-70">{cls.code}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="font-pixel text-xs opacity-80">
                You haven't joined any classrooms yet.
              </p>
            )}
          </PixelCard>

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
                  className="w-80 h-64 flex flex-col items-center justify-center gap-4 p-8 text-black"
                >
                  <img
                    src={item.icon}
                    alt={item.title}
                    className="h-24 w-auto select-none image-render-pixel"
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
    </div>
  );
}