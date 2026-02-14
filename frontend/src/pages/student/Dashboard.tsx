import { useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Logo } from "@/components/Logo";
import { PixelCard } from "@/components/PixelCard";
import { PixelButton } from "@/components/PixelButton";
import { Clock, HelpCircle, BookOpen, Gamepad2, Trophy, Target } from "lucide-react";
import learnIcon from "@/assets/file-logo.png";
import settingsIcon from "@/assets/settings-logo.png";
import playIcon from "@/assets/keyboard-logo.png";
import { resolveProfileImage } from "@/utils/profileAssets";
import fallbackProfilePic from "@/assets/cat-profile.jpg";
import bgVideo from "@/assets/b4.mp4";

export default function StudentDashboard() {
  const navigate = useNavigate();
  const userName = localStorage.getItem("userName") || "Student";
  const profileKey = localStorage.getItem("profilePicture") || "";
  const profilePic = resolveProfileImage(profileKey) || fallbackProfilePic;
  const [classrooms, setClassrooms] = useState<any[]>([]);
  const [activeEvalName, setActiveEvalName] = useState<string | null>(null);
  const [evalRemaining, setEvalRemaining] = useState(0);
  const [showHowToPlay, setShowHowToPlay] = useState(false);

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

    // Check for active evaluation
    const checkEval = async () => {
      const token = localStorage.getItem("token");
      if (!token) return;
      try {
        const BASE_URL = import.meta.env.VITE_API_URL.replace("/api/auth", "");
        const res = await fetch(`${BASE_URL}/api/student/evaluation-status`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (res.ok) {
          const data = await res.json();
          if (data.hasActiveEvaluation && data.evaluation) {
            setActiveEvalName(data.evaluation.classroomName);
            setEvalRemaining(data.evaluation.remainingSeconds);
          }
        }
      } catch { /* silent */ }
    };
    checkEval();
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

          {/* Active Activity / Graded Banner */}
          {activeEvalName && evalRemaining > 0 && (
            <PixelCard variant="red" className="text-white mb-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Clock size={20} />
                  <div>
                    <p className="font-pixel text-sm">
                      Active Activity — {activeEvalName}
                    </p>
                    <p className="font-pixel text-[10px] opacity-80">
                      {Math.ceil(evalRemaining / 60)} minutes remaining
                    </p>
                  </div>
                </div>
                <PixelButton variant="accent" size="sm" onClick={() => navigate("/student/play")}>
                  Go to Activity
                </PixelButton>
              </div>
            </PixelCard>
          )}

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
                    <span>
                      {cls.name}
                      {cls.teacher?.name && (
                        <span className="opacity-70 text-xs ml-2">| Teacher: {cls.teacher.name}</span>
                      )}
                    </span>
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

          {/* How to Play Button - Bottom Center */}
          <div className="flex justify-center mt-12">
            <PixelButton
              variant="secondary"
              size="lg"
              onClick={() => setShowHowToPlay(true)}
              className="flex items-center gap-2 shadow-lg"
            >
              How to Play
            </PixelButton>
          </div>
        </div>
      </div>

      {/* How to Play Modal */}
      {showHowToPlay && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 backdrop-blur-sm p-4 overflow-y-auto">
          <PixelCard className="max-w-4xl w-full my-8">
            <div className="p-8 max-h-[85vh] overflow-y-auto">

              {/* Header */}
              <div className="text-center mb-8">
                <h2 className="font-pixel text-3xl mb-3 text-purple-600">
                  How to Play TyPaw
                </h2>
                <p className="font-pixel text-sm text-muted-foreground">
                  Master touch typing with deep learning and hand tracking
                </p>
              </div>

              {/* Quick Start Guide */}
              <div className="bg-gradient-to-r from-yellow-500/20 to-orange-500/20 border-2 border-yellow-500/50 rounded-lg p-6 mb-8">
                <h3 className="font-pixel text-lg text-yellow-600 mb-4 flex items-center gap-2">
                  <Target size={20} />
                  Quick Start Guide
                </h3>
                <ol className="space-y-3">
                  <li className="font-pixel text-sm text-foreground">
                    <span className="inline-block bg-yellow-500 text-white rounded-full w-6 h-6 text-center mr-2">1</span>
                    <strong>Set up your camera</strong> in a top-view position above your keyboard
                  </li>
                  <li className="font-pixel text-sm text-foreground">
                    <span className="inline-block bg-yellow-500 text-white rounded-full w-6 h-6 text-center mr-2">2</span>
                    <strong>Start with Learn mode</strong> to master basic finger positions
                  </li>
                  <li className="font-pixel text-sm text-foreground">
                    <span className="inline-block bg-yellow-500 text-white rounded-full w-6 h-6 text-center mr-2">3</span>
                    <strong>Practice in Play mode</strong> to improve speed and accuracy
                  </li>
                  <li className="font-pixel text-sm text-foreground">
                    <span className="inline-block bg-yellow-500 text-white rounded-full w-6 h-6 text-center mr-2">4</span>
                    <strong>Take Evaluated sessions</strong> when your teacher assigns them
                  </li>
                </ol>
              </div>

              {/* Learn Mode Section */}
              <div className="mb-8">
                <div className="flex items-center gap-3 mb-4">
                  <div className="bg-yellow-500 rounded-lg p-3">
                    <BookOpen size={28} className="text-white" />
                  </div>
                  <div>
                    <h3 className="font-pixel text-xl text-yellow-600">Learn Mode</h3>
                    <p className="font-pixel text-xs text-muted-foreground">Master touch typing step-by-step</p>
                  </div>
                </div>

                <div className="bg-yellow-500/10 border-2 border-yellow-500/30 rounded-lg p-5 space-y-3">
                  <div>
                    <p className="font-pixel text-sm text-foreground font-semibold mb-2">📚 What is Learn Mode?</p>
                    <p className="font-pixel text-xs text-muted-foreground leading-relaxed">
                      Learn mode teaches you proper touch typing technique through progressive modules.
                      Each module focuses on specific keys and ensures you use the correct fingers.
                    </p>
                  </div>

                  <div>
                    <p className="font-pixel text-sm text-foreground font-semibold mb-2">🎯 How it Works:</p>
                    <ul className="font-pixel text-xs text-muted-foreground space-y-1.5 ml-4">
                      <li>• <strong>5 Progressive Modules:</strong> Home Row → Top Row → Bottom Row → Full Alphabet → Words</li>
                      <li>• <strong>Unlock System:</strong> Complete each module to unlock the next one</li>
                      <li>• <strong>Strict Validation:</strong> You must use the correct key AND correct finger to advance</li>
                      <li>• <strong>Real-time Feedback:</strong> Visual guides show proper finger placement for each key</li>
                      <li>• <strong>Progress Tracking:</strong> Your completion status is saved automatically</li>
                    </ul>
                  </div>

                  <div>
                    <p className="font-pixel text-sm text-foreground font-semibold mb-2">💡 Tips:</p>
                    <ul className="font-pixel text-xs text-muted-foreground space-y-1 ml-4">
                      <li>• Start with Module 1 (Home Row) even if you know how to type</li>
                      <li>• Focus on accuracy over speed - wrong fingers won't let you progress</li>
                      <li>• Use the keyboard images to memorize proper finger positions</li>
                    </ul>
                  </div>
                </div>
              </div>

              {/* Play Mode Section */}
              <div className="mb-8">
                <div className="flex items-center gap-3 mb-4">
                  <div className="bg-orange-500 rounded-lg p-3">
                    <Gamepad2 size={28} className="text-white" />
                  </div>
                  <div>
                    <h3 className="font-pixel text-xl text-orange-600">Play Mode</h3>
                    <p className="font-pixel text-xs text-muted-foreground">Practice and improve your skills</p>
                  </div>
                </div>

                <div className="bg-orange-500/10 border-2 border-orange-500/30 rounded-lg p-5 space-y-3">
                  <div>
                    <p className="font-pixel text-sm text-foreground font-semibold mb-2">🎮 What is Play Mode?</p>
                    <p className="font-pixel text-xs text-muted-foreground leading-relaxed">
                      Play mode lets you practice typing with real-time AI hand tracking.
                      Choose between Practice (for yourself) or Evaluated (graded by teacher).
                    </p>
                  </div>

                  <div>
                    <p className="font-pixel text-sm text-foreground font-semibold mb-2">🎯 How it Works:</p>
                    <ul className="font-pixel text-xs text-muted-foreground space-y-1.5 ml-4">
                      <li>• <strong>30-Second Timer:</strong> Type as much as you can in 30 seconds</li>
                      <li>• <strong>AI Hand Tracking:</strong> Camera detects which fingers you use for each key</li>
                      <li>• <strong>Real-time Metrics:</strong> See your WPM, accuracy, and errors as you type</li>
                      <li>• <strong>Detailed Feedback:</strong> Review your session with keystroke-by-keystroke analysis</li>
                      <li>• <strong>Difficulty Levels:</strong> Choose from Level 1 (easy) to Level 5 (advanced)</li>
                    </ul>
                  </div>
                </div>
              </div>

              {/* Practice vs Evaluated */}
              <div className="mb-8">
                <h3 className="font-pixel text-lg text-center mb-4 text-purple-600">
                  Practice vs Evaluated Sessions
                </h3>

                <div className="grid md:grid-cols-2 gap-4">
                  {/* Practice Mode */}
                  <div className="bg-blue-500/10 border-2 border-blue-500/40 rounded-lg p-5">
                    <div className="flex items-center gap-2 mb-3">
                      <Gamepad2 size={20} className="text-blue-500" />
                      <h4 className="font-pixel text-base text-blue-600 font-bold">Practice Mode</h4>
                    </div>
                    <ul className="font-pixel text-xs text-muted-foreground space-y-2">
                      <li>✅ <strong>For personal improvement</strong></li>
                      <li>✅ <strong>Not graded</strong> - results are private</li>
                      <li>✅ <strong>Unlimited attempts</strong> - practice as much as you want</li>
                      <li>✅ <strong>Choose any level</strong> - experiment freely</li>
                      <li>✅ <strong>Track your progress</strong> - see your improvement over time</li>
                    </ul>
                    <div className="mt-4 pt-3 border-t border-blue-500/30">
                      <p className="font-pixel text-[10px] text-blue-600 italic">
                        💡 Use Practice to warm up before Evaluated sessions!
                      </p>
                    </div>
                  </div>

                  {/* Evaluated Mode */}
                  <div className="bg-red-500/10 border-2 border-red-500/40 rounded-lg p-5">
                    <div className="flex items-center gap-2 mb-3">
                      <Trophy size={20} className="text-red-500" />
                      <h4 className="font-pixel text-base text-red-600 font-bold">Evaluated Mode</h4>
                    </div>
                    <ul className="font-pixel text-xs text-muted-foreground space-y-2">
                      <li>🏆 <strong>Graded by teacher</strong></li>
                      <li>🏆 <strong>Assigned by teacher</strong> - appears when active</li>
                      <li>🏆 <strong>Time-limited</strong> - must complete within deadline</li>
                      <li>🏆 <strong>Fixed difficulty</strong> - teacher chooses the level</li>
                      <li>🏆 <strong>Counts toward grade</strong> - affects your classroom score</li>
                    </ul>
                    <div className="mt-4 pt-3 border-t border-red-500/30">
                      <p className="font-pixel text-[10px] text-red-600 italic">
                        ⚠️ Active evaluations appear as a banner on your dashboard!
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Scoring System */}
              <div className="mb-8">
                <h3 className="font-pixel text-lg text-center mb-4 text-green-600">
                  How You're Scored
                </h3>

                <div className="bg-green-500/10 border-2 border-green-500/30 rounded-lg p-5">
                  <p className="font-pixel text-sm text-foreground font-semibold mb-3">📊 Your performance is measured by:</p>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <p className="font-pixel text-xs text-green-600 font-bold mb-1">Speed Metrics:</p>
                      <ul className="font-pixel text-xs text-muted-foreground space-y-1 ml-4">
                        <li>• <strong>Gross WPM:</strong> Total words typed per minute</li>
                        <li>• <strong>Net WPM:</strong> WPM minus errors (your true speed)</li>
                      </ul>
                    </div>
                    <div>
                      <p className="font-pixel text-xs text-green-600 font-bold mb-1">Accuracy Metrics:</p>
                      <ul className="font-pixel text-xs text-muted-foreground space-y-1 ml-4">
                        <li>• <strong>Key Accuracy:</strong> Correct keys pressed</li>
                        <li>• <strong>Finger Accuracy:</strong> Correct fingers used</li>
                        <li>• <strong>Error Rate:</strong> Percentage of mistakes</li>
                      </ul>
                    </div>
                  </div>
                  <div className="mt-4 pt-4 border-t border-green-500/30">
                    <p className="font-pixel text-xs text-foreground font-semibold mb-2">🎯 Final Grade (A-E):</p>
                    <p className="font-pixel text-xs text-muted-foreground leading-relaxed">
                      Your composite score combines speed, accuracy, and proper finger technique.
                      Using the correct fingers is just as important as typing fast!
                    </p>
                  </div>
                </div>
              </div>

              {/* Camera Setup Reminder */}
              <div className="bg-purple-500/10 border-2 border-purple-500/40 rounded-lg p-5 mb-6">
                <h4 className="font-pixel text-sm text-purple-600 font-bold mb-3">📷 Camera Setup Reminder:</h4>
                <ul className="font-pixel text-xs text-muted-foreground space-y-1.5 ml-4">
                  <li>• Position camera in <strong>top-view</strong> above your keyboard</li>
                  <li>• Ensure <strong>all keys are visible</strong> in the camera view</li>
                  <li>• <strong>Don't move</strong> the keyboard or camera during sessions</li>
                  <li>• Remove hands during initialization, then place them on home row</li>
                  <li>• Use good lighting to help the AI detect your hands accurately</li>
                </ul>
              </div>

              {/* Close Button */}
              <div className="flex justify-center">
                <PixelButton
                  variant="accent"
                  size="lg"
                  onClick={() => setShowHowToPlay(false)}
                  className="px-12"
                >
                  Got it! Let's Start
                </PixelButton>
              </div>
            </div>
          </PixelCard>
        </div>
      )}
    </div>
  );
}