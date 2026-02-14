import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Logo } from "@/components/Logo";
import { PixelButton } from "@/components/PixelButton";
import { PixelCard } from "@/components/PixelCard";
import { ArrowLeft, CheckCircle2, Lock } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import bgVideo from "@/assets/b4.mp4";

const modules = [
  {
    id: 1,
    title: "Home Row Heroes",
    description: "Master the foundation keys of touch typing.",
    focusKeys: "A S D F J K L",
  },
  {
    id: 2,
    title: "Top Row Adventure",
    description: "Reach up and conquer the top row.",
    focusKeys: "Q W E R T Y U I O P",
  },
  {
    id: 3,
    title: "Bottom Row Explorer",
    description: "Stretch down to complete the alphabet rows.",
    focusKeys: "Z X C V B N M",
  },
  {
    id: 4,
    title: "Alphabet Mastery",
    description: "Random letters from the entire keyboard.",
    focusKeys: "A – Z",
  },
  {
    id: 5,
    title: "Word Builder",
    description: "Type real words to build fluency.",
    focusKeys: "Full Words",
  },
];

const DEFAULT_PROGRESS: Record<number, { completed: boolean }> = {
  1: { completed: false },
  2: { completed: false },
  3: { completed: false },
  4: { completed: false },
  5: { completed: false },
};

const BASE_URL = import.meta.env.VITE_API_URL.replace("/api/auth", "");

export default function Learn() {
  const navigate = useNavigate();
  const [progress, setProgress] = useState<Record<number, { completed: boolean }>>(DEFAULT_PROGRESS);
  const [loading, setLoading] = useState(true);

  // Fetch progress from backend on mount
  useEffect(() => {
    const fetchProgress = async () => {
      const token = localStorage.getItem("token");
      if (!token) {
        setLoading(false);
        return;
      }

      try {
        const res = await fetch(`${BASE_URL}/api/student/learning-progress`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        const data = await res.json();
        if (res.ok) {
          // Convert API response to expected format
          const formattedProgress: Record<number, { completed: boolean }> = {};
          Object.keys(DEFAULT_PROGRESS).forEach((key) => {
            const moduleId = parseInt(key);
            formattedProgress[moduleId] = {
              completed: data[key]?.completed || false,
            };
          });
          setProgress(formattedProgress);
          // Also update localStorage as cache
          localStorage.setItem("typingModuleProgress", JSON.stringify(formattedProgress));
        } else {
          // Fallback to localStorage if API fails
          const saved = localStorage.getItem("typingModuleProgress");
          if (saved) {
            try {
              const parsed = JSON.parse(saved);
              if (parsed[6] !== undefined) {
                setProgress({ ...DEFAULT_PROGRESS });
              } else {
                setProgress(parsed);
              }
            } catch {
              setProgress({ ...DEFAULT_PROGRESS });
            }
          }
        }
      } catch (error) {
        console.error("Error fetching learning progress:", error);
        // Fallback to localStorage
        const saved = localStorage.getItem("typingModuleProgress");
        if (saved) {
          try {
            const parsed = JSON.parse(saved);
            if (parsed[6] !== undefined) {
              setProgress({ ...DEFAULT_PROGRESS });
            } else {
              setProgress(parsed);
            }
          } catch {
            setProgress({ ...DEFAULT_PROGRESS });
          }
        }
      } finally {
        setLoading(false);
      }
    };
    fetchProgress();
  }, []);

  const dynamicModules = modules.map((module) => {
    const prevId = module.id - 1;
    const isUnlocked =
      module.id === 1 || progress[prevId]?.completed === true;
    const isCompleted = progress[module.id]?.completed === true;
    return { ...module, unlocked: isUnlocked, completed: isCompleted };
  });

  const handleStartModule = (module: (typeof dynamicModules)[number]) => {
    if (!module.unlocked) return;
    navigate(`/student/learn/session?module=${module.id}`);
  };

  const handleResetProgress = async () => {
    if (!confirm("Reset all progress? This will lock all modules.")) return;

    const token = localStorage.getItem("token");
    if (token) {
      try {
        await fetch(`${BASE_URL}/api/student/learning-progress/reset`, {
          method: "PUT",
          headers: { Authorization: `Bearer ${token}` },
        });
      } catch (error) {
        console.error("Error resetting progress on backend:", error);
      }
    }

    // Reset local state and localStorage
    setProgress({ ...DEFAULT_PROGRESS });
    localStorage.removeItem("typingModuleProgress");
  };

  return (
    <div className="relative min-h-screen overflow-hidden">
      <video
        autoPlay
        loop
        muted
        playsInline
        className="absolute top-0 left-0 w-full h-full object-cover -z-10"
      >
        <source src={bgVideo} type="video/mp4" />
      </video>

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

          {/* Modules Grid - wrapped with readable background */}
          <div className="bg-black/50 backdrop-blur-sm rounded-lg p-6 border-2 border-yellow-300/30">
            {loading ? (
              <div className="text-center py-12">
                <p className="font-pixel text-lg text-white">Loading your progress...</p>
              </div>
            ) : (
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                {dynamicModules.map((module) => (
                <TooltipProvider key={module.id}>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <div>
                        <PixelCard
                          variant={module.unlocked ? "yellow" : "default"}
                          className={`relative transition-all duration-200 ${
                            module.unlocked
                              ? "cursor-pointer hover:brightness-110"
                              : "opacity-50 cursor-not-allowed"
                          }`}
                        >
                          <div className="flex items-start justify-between mb-2">
                            <h3 className="font-pixel text-lg text-black">
                              {module.title}
                            </h3>
                            {module.completed && (
                              <CheckCircle2
                                className="text-green-600 drop-shadow-sm"
                                size={24}
                              />
                            )}
                            {!module.unlocked && (
                              <Lock className="text-black/60" size={24} />
                            )}
                          </div>

                          <p className="font-pixel text-xs text-black/70 mb-1">
                            {module.focusKeys}
                          </p>

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
                      </div>
                    </TooltipTrigger>
                    {!module.unlocked && (
                      <TooltipContent className="font-pixel text-xs bg-black/90 text-yellow-300 border-2 border-yellow-400">
                        <p>🔒 Complete Module {module.id - 1} first to unlock this module</p>
                      </TooltipContent>
                    )}
                  </Tooltip>
                </TooltipProvider>
              ))}
              </div>
            )}
          </div>

          {/* Reset Progress */}
          <div className="mt-12 text-center opacity-70">
            <PixelButton
              variant="secondary"
              size="sm"
              onClick={handleResetProgress}
            >
              Reset Progress
            </PixelButton>
          </div>
        </div>
      </div>
    </div>
  );
}
