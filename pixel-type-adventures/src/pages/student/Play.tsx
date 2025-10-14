import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Logo } from "@/components/Logo";
import { PixelButton } from "@/components/PixelButton";
import { PixelCard } from "@/components/PixelCard";
import { ArrowLeft, Trophy, Clock } from "lucide-react";

export default function Play() {
  const navigate = useNavigate();
  const [sessionType, setSessionType] = useState<"practice" | "evaluated">("practice");
  const [level, setLevel] = useState(1);

  const levels = [
    { id: 1, name: "Level 1", description: "Letters only" },
    { id: 2, name: "Level 2", description: "Random words" },
    { id: 3, name: "Level 3", description: "Phrases with numbers" },
    { id: 4, name: "Level 4", description: "Full paragraphs" },
  ];

  const history = [
    { date: "2024-01-15", level: 1, wpm: 25, accuracy: 92, grade: "A" },
    { date: "2024-01-14", level: 1, wpm: 22, accuracy: 88, grade: "B+" },
    { date: "2024-01-13", level: 1, wpm: 20, accuracy: 85, grade: "B" },
  ];

  const startSession = () => {
    navigate(`/student/play/session?type=${sessionType}&level=${level}`);
  };

  return (
    <div className="min-h-screen p-8">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center gap-4 mb-12">
          <PixelButton variant="secondary" onClick={() => navigate("/student/dashboard")}>
            <ArrowLeft size={20} />
          </PixelButton>
          <Logo />
        </div>

        <div className="grid lg:grid-cols-2 gap-8 mb-8">
          {/* Session Setup */}
          <PixelCard variant="orange" className="text-white">
            <h2 className="font-pixel text-xl mb-6">Start Session</h2>

            <div className="space-y-6">
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
                  {sessionType === "practice" ? "Not graded, for practice" : "Graded session"}
                </p>
              </div>

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

              <PixelButton variant="accent" onClick={startSession} className="w-full" size="lg">
                Start Typing!
              </PixelButton>
            </div>
          </PixelCard>

          {/* Stats Preview */}
          <PixelCard variant="purple" className="text-white">
            <h2 className="font-pixel text-xl mb-6">Your Stats</h2>
            <div className="space-y-4">
              <div className="flex items-center gap-4">
                <Trophy size={32} />
                <div>
                  <p className="font-pixel text-xs opacity-90">Best WPM</p>
                  <p className="font-pixel text-2xl">25</p>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <Clock size={32} />
                <div>
                  <p className="font-pixel text-xs opacity-90">Sessions</p>
                  <p className="font-pixel text-2xl">{history.length}</p>
                </div>
              </div>
            </div>
          </PixelCard>
        </div>

        {/* History */}
        <PixelCard>
          <h2 className="font-pixel text-xl mb-6">Recent Sessions</h2>
          <div className="space-y-3">
            {history.map((session, idx) => (
              <div
                key={idx}
                className="pixel-border p-4 bg-muted flex items-center justify-between"
              >
                <div>
                  <p className="font-pixel text-sm">Level {session.level}</p>
                  <p className="font-pixel text-xs text-muted-foreground">{session.date}</p>
                </div>
                <div className="text-right">
                  <p className="font-pixel text-sm">{session.wpm} WPM</p>
                  <p className="font-pixel text-xs text-muted-foreground">
                    {session.accuracy}% Accuracy
                  </p>
                </div>
                <div className="font-pixel text-xl text-primary">{session.grade}</div>
              </div>
            ))}
          </div>
        </PixelCard>
      </div>
    </div>
  );
}
