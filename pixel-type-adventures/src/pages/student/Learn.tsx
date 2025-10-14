import { useNavigate } from "react-router-dom";
import { Logo } from "@/components/Logo";
import { PixelButton } from "@/components/PixelButton";
import { PixelCard } from "@/components/PixelCard";
import { ArrowLeft, CheckCircle2, Lock } from "lucide-react";

export default function Learn() {
  const navigate = useNavigate();

  const modules = [
    {
      id: 1,
      title: "Home Row Keys",
      description: "Learn ASDF JKL;",
      unlocked: true,
      completed: false,
    },
    {
      id: 2,
      title: "Top Row Keys",
      description: "Master QWER UIOP",
      unlocked: true,
      completed: false,
    },
    {
      id: 3,
      title: "Bottom Row Keys",
      description: "Practice ZXCV NM,.",
      unlocked: false,
      completed: false,
    },
    {
      id: 4,
      title: "Numbers Row",
      description: "Learn 1234567890",
      unlocked: false,
      completed: false,
    },
    {
      id: 5,
      title: "Special Characters",
      description: "Master !@#$%^&*()",
      unlocked: false,
      completed: false,
    },
    {
      id: 6,
      title: "Full Keyboard",
      description: "Complete mastery",
      unlocked: false,
      completed: false,
    },
  ];

  return (
    <div className="min-h-screen p-8">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center justify-between mb-12">
          <div className="flex items-center gap-4">
            <PixelButton variant="secondary" onClick={() => navigate("/student/dashboard")}>
              <ArrowLeft size={20} />
            </PixelButton>
            <Logo />
          </div>
          <h1 className="font-pixel text-2xl">Learning Modules</h1>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {modules.map((module) => (
            <PixelCard
              key={module.id}
              variant={module.unlocked ? "yellow" : "default"}
              className={`relative ${module.unlocked ? "cursor-pointer hover:brightness-110" : "opacity-50"}`}
            >
              <div className="flex items-start justify-between mb-4">
                <h3 className="font-pixel text-lg">{module.title}</h3>
                {module.completed && <CheckCircle2 className="text-green-600" size={24} />}
                {!module.unlocked && <Lock size={24} />}
              </div>
              <p className="font-pixel text-xs text-muted-foreground mb-4">
                {module.description}
              </p>
              <PixelButton
                variant={module.unlocked ? "learn" : "primary"}
                size="sm"
                className="w-full"
                disabled={!module.unlocked}
              >
                {module.completed ? "Review" : "Start"}
              </PixelButton>
            </PixelCard>
          ))}
        </div>
      </div>
    </div>
  );
}
