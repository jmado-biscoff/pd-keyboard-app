import { useNavigate } from "react-router-dom";
import { Logo } from "@/components/Logo";
import { PixelButton } from "@/components/PixelButton";
import { PixelCard } from "@/components/PixelCard";
import { ArrowLeft, Trophy, Copy } from "lucide-react";
import { toast } from "sonner";

export default function Classroom() {
  const navigate = useNavigate();
  const classroomCode = "TYP-2024";

  const students = [
    { id: 1, name: "Alice Johnson", avatar: "🦁", wpm: 45, accuracy: 95, sessions: 24 },
    { id: 2, name: "Bob Smith", avatar: "🐯", wpm: 38, accuracy: 92, sessions: 20 },
    { id: 3, name: "Charlie Brown", avatar: "🐻", wpm: 52, accuracy: 97, sessions: 30 },
    { id: 4, name: "Diana Prince", avatar: "🦊", wpm: 41, accuracy: 93, sessions: 22 },
    { id: 5, name: "Ethan Hunt", avatar: "🐺", wpm: 35, accuracy: 88, sessions: 18 },
  ];

  const copyClassroomCode = () => {
    navigator.clipboard.writeText(classroomCode);
    toast.success("Classroom code copied!");
  };

  return (
    <div className="min-h-screen p-8">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center gap-4 mb-12">
          <PixelButton variant="secondary" onClick={() => navigate("/teacher/dashboard")}>
            <ArrowLeft size={20} />
          </PixelButton>
          <Logo />
        </div>

        {/* Classroom Code */}
        <PixelCard variant="orange" className="mb-8 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-pixel text-sm mb-2">Classroom Code</p>
              <p className="font-pixel text-3xl">{classroomCode}</p>
            </div>
            <PixelButton variant="accent" onClick={copyClassroomCode} className="flex items-center gap-2">
              <Copy size={20} />
              Copy
            </PixelButton>
          </div>
        </PixelCard>

        {/* Leaderboard */}
        <PixelCard variant="yellow" className="mb-8">
          <h2 className="font-pixel text-xl mb-6 flex items-center gap-2">
            <Trophy size={24} className="text-primary" />
            Leaderboard
          </h2>
          <div className="space-y-3">
            {students
              .sort((a, b) => b.wpm - a.wpm)
              .map((student, idx) => (
                <div
                  key={student.id}
                  className="pixel-border p-4 bg-background flex items-center gap-4"
                >
                  <div className="font-pixel text-2xl w-12 text-center">
                    {idx === 0 ? "🥇" : idx === 1 ? "🥈" : idx === 2 ? "🥉" : `#${idx + 1}`}
                  </div>
                  <div className="text-4xl">{student.avatar}</div>
                  <div className="flex-1">
                    <p className="font-pixel text-sm">{student.name}</p>
                    <p className="font-pixel text-xs text-muted-foreground">
                      {student.sessions} sessions
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="font-pixel text-lg text-primary">{student.wpm} WPM</p>
                    <p className="font-pixel text-xs text-muted-foreground">
                      {student.accuracy}% accuracy
                    </p>
                  </div>
                </div>
              ))}
          </div>
        </PixelCard>

        {/* Student List */}
        <PixelCard>
          <h2 className="font-pixel text-xl mb-6">All Students ({students.length})</h2>
          <div className="grid md:grid-cols-2 gap-4">
            {students.map((student) => (
              <div key={student.id} className="pixel-border p-4 bg-muted">
                <div className="flex items-center gap-3 mb-3">
                  <div className="text-3xl">{student.avatar}</div>
                  <p className="font-pixel text-sm">{student.name}</p>
                </div>
                <div className="grid grid-cols-3 gap-2">
                  <div>
                    <p className="font-pixel text-[10px] text-muted-foreground">WPM</p>
                    <p className="font-pixel text-sm">{student.wpm}</p>
                  </div>
                  <div>
                    <p className="font-pixel text-[10px] text-muted-foreground">Accuracy</p>
                    <p className="font-pixel text-sm">{student.accuracy}%</p>
                  </div>
                  <div>
                    <p className="font-pixel text-[10px] text-muted-foreground">Sessions</p>
                    <p className="font-pixel text-sm">{student.sessions}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </PixelCard>
      </div>
    </div>
  );
}
