import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Logo } from "@/components/Logo";
import { PixelButton } from "@/components/PixelButton";
import { PixelCard } from "@/components/PixelCard";
import { ArrowLeft, Trophy, Clock, RefreshCw } from "lucide-react";
import bgVideo from "@/assets/bg1.mp4";

const BASE_URL = import.meta.env.VITE_API_URL.replace("/api/auth", "");

export default function Play() {
  const navigate = useNavigate();
  const [sessionType, setSessionType] = useState<"practice" | "evaluated">("practice");
  const [level, setLevel] = useState(1);
  const [history, setHistory] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const levels = [
    { id: 1, name: "Level 1", description: "50 random characters" },
    { id: 2, name: "Level 2", description: "50-character word sequence" },
  ];

  // 🔹 Fetch recent sessions from MongoDB (Privacy filtered)
  const fetchResults = async () => {
    setIsLoading(true);
    const userId = localStorage.getItem("userName") || "guest";

    try {
      const res = await fetch(`${BASE_URL}/api/results?userId=${userId}`);
      if (!res.ok) throw new Error("Failed to fetch");
      const data = await res.json();
      setHistory(data);
    } catch (err) {
      console.error("❌ Failed to fetch results:", err);
      setHistory([]); // Set empty array if backend unavailable
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch on mount
  useEffect(() => {
    fetchResults();
  }, []);

  // Refresh data when window gains focus (user returns from typing session)
  useEffect(() => {
    const handleFocus = () => {
      console.log("🔄 Window focused - refreshing dashboard data");
      fetchResults();
    };

    window.addEventListener("focus", handleFocus);
    return () => window.removeEventListener("focus", handleFocus);
  }, []);

  // 🔹 Start typing session
  const startSession = () => {
    navigate(`/student/play/session?type=${sessionType}&level=${level}`);
  };

  // 🔹 Compute best WPM for Stats card
  const bestWPM = history.length > 0 ? Math.max(...history.map((s) => s.wpm || 0)) : 0;

  return (
    <div className="relative h-screen overflow-hidden">
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
      <div className="relative z-10 p-8 h-screen overflow-y-auto text-white pixel-scrollbar-vertical">
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

            {/* 🔸 Last Session Performance */}
            <PixelCard
              variant="purple"
              className="text-white bg-black/60 border-[3px] border-purple-400 backdrop-blur-sm"
            >
              <div className="flex items-center justify-between mb-6">
                <h2 className="font-pixel text-xl">Last Session Performance</h2>
                <button
                  onClick={fetchResults}
                  disabled={isLoading}
                  className="p-2 hover:bg-purple-500/20 rounded transition-colors disabled:opacity-50"
                  title="Refresh data"
                >
                  <RefreshCw size={20} className={`text-purple-300 ${isLoading ? 'animate-spin' : ''}`} />
                </button>
              </div>

              {isLoading ? (
                // Loading state
                <div className="flex flex-col items-center justify-center py-8">
                  <RefreshCw size={48} className="text-purple-300 mb-4 animate-spin" />
                  <p className="font-pixel text-sm text-center">
                    Syncing Performance Data...
                  </p>
                </div>
              ) : history.length === 0 ? (
                // Empty state
                <div className="flex flex-col items-center justify-center py-8 opacity-50">
                  <Trophy size={48} className="text-purple-300 mb-4" />
                  <p className="font-pixel text-sm text-center">
                    No sessions yet.<br />Start typing to see your stats!
                  </p>
                </div>
              ) : (
                <div className="space-y-6">
                  {/* Summary Graph - Stacked Bar */}
                  <div>
                    <p className="font-pixel text-[10px] uppercase tracking-widest text-muted-foreground/50 mb-2">
                      Keystroke Breakdown
                    </p>
                    <div className="h-8 flex rounded-md overflow-hidden border-2 border-white/30">
                      {(() => {
                        const lastSession = history[0];
                        const correctCount = lastSession.correctCount || 0;
                        const wrongKeys = lastSession.wrongKeysCount || 0;
                        const wrongFingers = lastSession.wrongFingersCount || 0;
                        const skipped = lastSession.skippedCount || 0;
                        const total = correctCount + wrongKeys + wrongFingers + skipped;

                        const correctPct = total > 0 ? (correctCount / total) * 100 : 0;
                        const wrongKeyPct = total > 0 ? (wrongKeys / total) * 100 : 0;
                        const wrongFingerPct = total > 0 ? (wrongFingers / total) * 100 : 0;
                        const skippedPct = total > 0 ? (skipped / total) * 100 : 0;

                        return (
                          <>
                            {correctPct > 0 && (
                              <div
                                className="bg-green-500 flex items-center justify-center text-white font-pixel text-[9px]"
                                style={{ width: `${correctPct}%` }}
                                title={`Correct: ${correctCount}`}
                              >
                                {correctPct > 15 && correctCount}
                              </div>
                            )}
                            {wrongFingerPct > 0 && (
                              <div
                                className="bg-orange-500 flex items-center justify-center text-white font-pixel text-[9px]"
                                style={{ width: `${wrongFingerPct}%` }}
                                title={`Wrong Finger: ${wrongFingers}`}
                              >
                                {wrongFingerPct > 15 && wrongFingers}
                              </div>
                            )}
                            {wrongKeyPct > 0 && (
                              <div
                                className="bg-red-500 flex items-center justify-center text-white font-pixel text-[9px]"
                                style={{ width: `${wrongKeyPct}%` }}
                                title={`Wrong Key: ${wrongKeys}`}
                              >
                                {wrongKeyPct > 15 && wrongKeys}
                              </div>
                            )}
                            {skippedPct > 0 && (
                              <div
                                className="bg-black flex items-center justify-center text-white font-pixel text-[9px]"
                                style={{ width: `${skippedPct}%` }}
                                title={`Skipped: ${skipped}`}
                              >
                                {skippedPct > 15 && skipped}
                              </div>
                            )}
                          </>
                        );
                      })()}
                    </div>
                    {/* Legend */}
                    <div className="grid grid-cols-2 gap-2 mt-3">
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 bg-green-500 rounded"></div>
                        <span className="font-pixel text-[9px] text-white/80">Correct</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 bg-orange-500 rounded"></div>
                        <span className="font-pixel text-[9px] text-white/80">Wrong Finger</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 bg-red-500 rounded"></div>
                        <span className="font-pixel text-[9px] text-white/80">Wrong Key</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 bg-black border border-white/30 rounded"></div>
                        <span className="font-pixel text-[9px] text-white/80">Skipped</span>
                      </div>
                    </div>
                  </div>

                  {/* Metrics Grid */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-black/40 rounded-lg p-3 border border-white/20">
                      <p className="font-pixel text-[9px] uppercase text-purple-300 mb-1">Rating</p>
                      <p className="font-pixel text-4xl text-yellow-400">{history[0].grade || "---"}</p>
                    </div>
                    <div className="bg-black/40 rounded-lg p-3 border border-white/20">
                      <p className="font-pixel text-[9px] uppercase text-purple-300 mb-1">Net WPM</p>
                      <p className="font-pixel text-3xl text-white">
                        {history[0].netWpm != null ? Math.round(history[0].netWpm) : "---"}
                      </p>
                    </div>
                    <div className="bg-black/40 rounded-lg p-3 border border-white/20">
                      <p className="font-pixel text-[9px] uppercase text-purple-300 mb-1">Score</p>
                      <p className="font-pixel text-3xl text-white">
                        {history[0].compositeScore != null ? Number(history[0].compositeScore).toFixed(1) : "---"}
                      </p>
                    </div>
                    <div className="bg-black/40 rounded-lg p-3 border border-white/20">
                      <p className="font-pixel text-[9px] uppercase text-purple-300 mb-1">Error Rate</p>
                      <p className="font-pixel text-3xl text-white">
                        {history[0].errorRate != null ? `${Number(history[0].errorRate).toFixed(1)}%` : "---"}
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </PixelCard>
          </div>

          {/* 🔸 Recent Sessions History */}
          <PixelCard className="bg-black/60 border-[3px] border-yellow-300 backdrop-blur-sm">
            <h2 className="font-pixel text-xl mb-6 text-yellow-200">
              Recent Sessions
            </h2>
            {history.length === 0 ? (
              <p className="font-pixel text-sm text-gray-400 text-center py-8">
                No sessions recorded yet.
              </p>
            ) : (
              <div className="overflow-y-auto pixel-scrollbar-vertical pr-2" style={{ maxHeight: '220px' }}>
                <div className="space-y-3">
                  {history.map((session, idx) => (
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
                        <p className="font-pixel text-sm">{Math.round(session.wpm)} WPM</p>
                        <p className="font-pixel text-xs opacity-80">
                          {session.accuracy != null ? `${Number(session.accuracy).toFixed(1)}%` : "---"} Accuracy
                        </p>
                      </div>
                      <div className="font-pixel text-xl text-yellow-400">
                        {session.grade || "-"}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

          </PixelCard>
        </div>
      </div>
    </div>
  );
}
