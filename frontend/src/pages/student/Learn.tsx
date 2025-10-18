import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Logo } from "@/components/Logo";
import { PixelButton } from "@/components/PixelButton";
import { PixelCard } from "@/components/PixelCard";
import { ArrowLeft, CheckCircle2, Lock } from "lucide-react";
import bgVideo from "@/assets/b4.mp4";

export default function Learn() {
  const navigate = useNavigate();

  // 🔹 Initialize progress from localStorage or start fresh
  const [progress, setProgress] = useState(() => {
    const saved = localStorage.getItem("typingModuleProgress");
    return saved
      ? JSON.parse(saved)
      : {
          1: { completed: false },
          2: { completed: false },
          3: { completed: false },
          4: { completed: false },
          5: { completed: false },
          6: { completed: false },
        };
  });

  // 🔹 Save progress on update
  useEffect(() => {
    localStorage.setItem("typingModuleProgress", JSON.stringify(progress));
  }, [progress]);

  // 🔹 Modules configuration
  const modules = [
    {
      id: 1,
      title: "Home Row Keys",
      description: "Learn ASDF JKL;",
      path: "home-row",
    },
    {
      id: 2,
      title: "Top Row Keys",
      description: "Master QWER UIOP",
      path: "top-row",
    },
    {
      id: 3,
      title: "Bottom Row Keys",
      description: "Practice ZXCV NM,.",
      path: "bottom-row",
    },
    {
      id: 4,
      title: "Numbers Row",
      description: "Learn 1234567890",
      path: "numbers-row",
    },
    {
      id: 5,
      title: "Special Characters",
      description: "Master !@#$%^&*()",
      path: "special-characters",
    },
    {
      id: 6,
      title: "Full Keyboard",
      description: "Complete mastery",
      path: "full-keyboard",
    },
  ];

  // 🔹 Determine unlocks dynamically
  const getModules = () => {
    return modules.map((module) => {
      const prevId = module.id - 1;
      const isUnlocked =
        module.id === 1 || progress[prevId]?.completed === true;
      const isCompleted = progress[module.id]?.completed === true;
      return {
        ...module,
        unlocked: isUnlocked,
        completed: isCompleted,
      };
    });
  };

  const handleStartModule = (module) => {
    if (!module.unlocked) return;
    navigate(`/student/learn/${module.path}`);
  };

  const dynamicModules = getModules();

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
          <div className="flex items-center justify-between mb-12">
            <div className="flex items-center gap-4">
              <PixelButton
                variant="secondary"
                onClick={() => navigate("/student/dashboard")}
              >
                <ArrowLeft size={20} />
              </PixelButton>
              <Logo />
            </div>
            <h1 className="font-pixel text-2xl text-black">
              Learning Modules
            </h1>
          </div>

          {/* Modules Grid */}
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {dynamicModules.map((module) => (
              <PixelCard
                key={module.id}
                variant={module.unlocked ? "yellow" : "default"}
                className={`relative transition-all duration-200 ${
                  module.unlocked
                    ? "cursor-pointer hover:brightness-110"
                    : "opacity-50 cursor-not-allowed"
                }`}
              >
                <div className="flex items-start justify-between mb-4">
                  <h3 className="font-pixel text-lg text-black">
                    {module.title}
                  </h3>
                  {module.completed && (
                    <CheckCircle2
                      className="text-green-600 drop-shadow-sm"
                      size={24}
                    />
                  )}
                  {!module.unlocked && <Lock className="text-black/60" size={24} />}
                </div>

                <p className="font-pixel text-xs text-black mb-4">
                  {module.description}
                </p>

                <PixelButton
                  variant={module.unlocked ? "learn" : "primary"}
                  size="sm"
                  className="w-full"
                  disabled={!module.unlocked}
                  onClick={() => handleStartModule(module)}
                >
                  {module.completed ? "Review" : "Start"}
                </PixelButton>
              </PixelCard>
            ))}
          </div>

          {/* Optional Developer Controls */}
          <div className="mt-12 text-center opacity-70">
            <PixelButton
              variant="secondary"
              size="sm"
              onClick={() => {
                if (
                  confirm("Reset all progress? This will lock all modules.")
                ) {
                  localStorage.removeItem("typingModuleProgress");
                  window.location.reload();
                }
              }}
            >
              Reset Progress
            </PixelButton>
          </div>
        </div>
      </div>
    </div>
  );
}
