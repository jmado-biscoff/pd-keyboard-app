import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Logo } from "@/components/Logo";
import { PixelButton } from "@/components/PixelButton";
import { PixelCard } from "@/components/PixelCard";
import { ArrowLeft, Trophy, Clock } from "lucide-react";
import bgVideo from "@/assets/bg1.mp4";

export default function Play() {
  const navigate = useNavigate();
  const [sessionType, setSessionType] = useState<"practice" | "evaluated">("practice");
  const [level, setLevel] = useState(1);
  const [history, setHistory] = useState<any[]>([]);

  const levels = [
    { id: 1, name: "Level 1", description: "Letters only" },
    { id: 2, name: "Level 2", description: "Random words" },
    { id: 3, name: "Level 3", description: "Short phrases" },
    { id: 4, name: "Level 4", description: "Full sentences" },
  ];

  // 🔹 Local fallback data (if fetch fails)
  const fallbackHistory = [
    { date: "2024-01-15", level: 1, wpm: 25, accuracy: 92, grade: "A" },
    { date: "2024-01-14", level: 1, wpm: 22, accuracy: 88, grade: "B+" },
    { date: "2024-01-13", level: 1, wpm: 20, accuracy: 85, grade: "B" },
  ];

  // 🔹 Fetch recent sessions from MongoDB
  useEffect(() => {
    const fetchResults = async () => {
      try {
        const res = await fetch("http://localhost:5000/api/results");
        if (!res.ok) throw new Error("Failed to fetch");
        const data = await res.json();
        setHistory(data);
      } catch (err) {
        console.error("❌ Failed to fetch results:", err);
        setHistory(fallbackHistory); // Use local fallback if backend unavailable
      }
    };

    fetchResults();
  }, []);

  // 🔹 Start typing session
  const startSession = () => {
    navigate(`/student/play/session?type=${sessionType}&level=${level}`);
  };

  // 🔹 Compute best WPM for Stats card
  const bestWPM = history.length > 0 ? Math.max(...history.map((s) => s.wpm || 0)) : 0;

  return (
    <div className="relative min-h-screen overflow-hidden">
      {/* 🔹 Background video */}
      <video
        autoPlay
        loop
        muted
        playsInline
        className="absolute top-0 left-0 w-full h-full object-cover -z-10"
      >
        <source src={bgVideo} type="video/mp4" />
      </video>

      {/* 🔹 Overlay */}
      <div className="absolute inset-0 bg-black/40 -z-10" />

      {/* 🔹 Page content */}
      <div className="relative z-10 p-8 min-h-screen text-white">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <div className="flex items-center gap-4 mb-12">
            <PixelButton
              variant="secondary"
              onClick={() => navigate("/student/dashboard")}
            >
              <ArrowLeft size={20} />
            </PixelButton>
            <Logo />
          </div>

          <div className="grid lg:grid-cols-2 gap-8 mb-8">
            {/* 🔸 Session Setup */}
            <PixelCard
              variant="orange"
              className="text-white bg-black/60 border-[3px] border-orange-400 backdrop-blur-sm"
            >
              <h2 className="font-pixel text-xl mb-6">Start Session</h2>

              <div className="space-y-6">
                {/* Session Type */}
                <div>
                  <label className="block font-pixel text-xs mb-3">Session Type</label>
                  <div className="flex gap-2">
                    <PixelButton
                      variant={sessionType === "practice" ? "accent" : "primary"}
                      onClick={() => setSessionType("practice")}
                      className="flex-1"
                    >
                      Practice
                    </PixelButton>
                    <PixelButton
                      variant={sessionType === "evaluated" ? "accent" : "primary"}
                      onClick={() => setSessionType("evaluated")}
                      className="flex-1"
                    >
                      Evaluated
                    </PixelButton>
                  </div>
                  <p className="font-pixel text-[10px] mt-2 opacity-90">
                    {sessionType === "practice"
                      ? "Not graded, perfect for drills"
                      : "Graded and recorded for progress"}
                  </p>
                </div>

                {/* Level Selection */}
                <div>
                  <label className="block font-pixel text-xs mb-3">Select Level</label>
                  <div className="grid grid-cols-2 gap-2">
                    {levels.map((lvl) => (
                      <PixelButton
                        key={lvl.id}
                        variant={level === lvl.id ? "accent" : "primary"}
                        onClick={() => setLevel(lvl.id)}
                        size="sm"
                      >
                        {lvl.name}
                      </PixelButton>
                    ))}
                  </div>
                  <p className="font-pixel text-[10px] mt-2 opacity-90">
                    {levels.find((l) => l.id === level)?.description}
                  </p>
                </div>

                <PixelButton
                  variant="accent"
                  onClick={startSession}
                  className="w-full"
                  size="lg"
                >
                  Start Typing!
                </PixelButton>
              </div>
            </PixelCard>

            {/* 🔸 Stats Preview */}
            <PixelCard
              variant="purple"
              className="text-white bg-black/60 border-[3px] border-purple-400 backdrop-blur-sm"
            >
              <h2 className="font-pixel text-xl mb-6">Your Stats</h2>
              <div className="space-y-4">
                <div className="flex items-center gap-4">
                  <Trophy size={32} className="text-yellow-300" />
                  <div>
                    <p className="font-pixel text-xs opacity-90">Best WPM</p>
                    <p className="font-pixel text-2xl">{bestWPM}</p>
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  <Clock size={32} className="text-blue-300" />
                  <div>
                    <p className="font-pixel text-xs opacity-90">Sessions</p>
                    <p className="font-pixel text-2xl">{history.length}</p>
                  </div>
                </div>
              </div>
            </PixelCard>
          </div>

          {/* 🔸 Recent Sessions History */}
          <PixelCard className="bg-black/60 border-[3px] border-yellow-300 backdrop-blur-sm">
            <h2 className="font-pixel text-xl mb-6 text-yellow-200">
              Recent Sessions
            </h2>
            <div className="space-y-3">
              {history.length === 0 ? (
                <p className="font-pixel text-sm text-gray-400 text-center">
                  No sessions recorded yet.
                </p>
              ) : (
                history.map((session, idx) => (
                  <div
                    key={idx}
                    className="pixel-border p-4 bg-black/50 flex items-center justify-between text-white border border-yellow-300 rounded-md"
                  >
                    <div>
                      <p className="font-pixel text-sm">Level {session.level}</p>
                      <p className="font-pixel text-xs opacity-80">
                        {new Date(session.date).toISOString().split("T")[0]}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="font-pixel text-sm">{session.wpm} WPM</p>
                      <p className="font-pixel text-xs opacity-80">
                        {session.accuracy}% Accuracy
                      </p>
                    </div>
                    <div className="font-pixel text-xl text-yellow-400">
                      {session.grade || "-"}
                    </div>
                  </div>
                ))
              )}
            </div>
          </PixelCard>
        </div>
      </div>
    </div>
  );
}
