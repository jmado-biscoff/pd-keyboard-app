import { useState, useEffect } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { Logo } from "@/components/Logo";
import { PixelButton } from "@/components/PixelButton";
import { PixelCard } from "@/components/PixelCard";
import { ArrowLeft } from "lucide-react";

export default function PlaySession() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const sessionType = searchParams.get("type") || "practice";
  const level = parseInt(searchParams.get("level") || "1");

  const [currentText] = useState("the quick brown fox jumps over the lazy dog");
  const [userInput, setUserInput] = useState("");
  const [wpm, setWpm] = useState(0);
  const [accuracy, setAccuracy] = useState(100);
  const [correctFingers, setCorrectFingers] = useState(0);
  const [incorrectFingers, setIncorrectFingers] = useState(0);
  const [startTime] = useState(Date.now());

  const keyboardLayout = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
    ["Z", "X", "C", "V", "B", "N", "M"],
  ];

  useEffect(() => {
    const timeElapsed = (Date.now() - startTime) / 1000 / 60;
    const wordsTyped = userInput.trim().split(" ").length;
    setWpm(Math.round(wordsTyped / timeElapsed) || 0);

    let correct = 0;
    for (let i = 0; i < userInput.length; i++) {
      if (userInput[i] === currentText[i]) correct++;
    }
    setAccuracy(userInput.length > 0 ? Math.round((correct / userInput.length) * 100) : 100);
  }, [userInput, startTime, currentText]);

  return (
    <div className="min-h-screen p-8">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <PixelButton variant="secondary" onClick={() => navigate("/student/play")}>
              <ArrowLeft size={20} />
            </PixelButton>
            <Logo />
          </div>
          <div className="font-pixel text-sm">
            {sessionType === "evaluated" ? "🏆 Graded Session" : "🎮 Practice Mode"} - Level {level}
          </div>
        </div>

        {/* Stats Bar */}
        <div className="grid grid-cols-2 lg:grid-cols-5 gap-4 mb-8">
          <PixelCard variant="yellow">
            <p className="font-pixel text-xs mb-1">WPM</p>
            <p className="font-pixel text-2xl">{wpm}</p>
          </PixelCard>
          <PixelCard variant="orange">
            <p className="font-pixel text-xs mb-1">Accuracy</p>
            <p className="font-pixel text-2xl">{accuracy}%</p>
          </PixelCard>
          <PixelCard className="bg-green-500 text-white">
            <p className="font-pixel text-xs mb-1">Correct</p>
            <p className="font-pixel text-2xl">{correctFingers}</p>
          </PixelCard>
          <PixelCard className="bg-red-500 text-white">
            <p className="font-pixel text-xs mb-1">Incorrect</p>
            <p className="font-pixel text-2xl">{incorrectFingers}</p>
          </PixelCard>
          <PixelCard variant="purple" className="text-white">
            <p className="font-pixel text-xs mb-1">Words</p>
            <p className="font-pixel text-2xl">{userInput.trim().split(" ").length}</p>
          </PixelCard>
        </div>

        {/* Text Display */}
        <PixelCard className="mb-8">
          <div className="font-pixel text-lg leading-relaxed mb-4">
            {currentText.split("").map((char, idx) => {
              let color = "text-muted-foreground";
              if (idx < userInput.length) {
                color = userInput[idx] === char ? "text-green-600" : "text-red-600";
              } else if (idx === userInput.length) {
                color = "text-primary animate-pulse";
              }
              return (
                <span key={idx} className={color}>
                  {char}
                </span>
              );
            })}
          </div>
          <input
            type="text"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            className="w-full px-4 py-3 bg-input border-[3px] border-border text-foreground font-pixel text-sm focus:outline-none focus:ring-2 focus:ring-primary"
            placeholder="Start typing..."
            autoFocus
          />
        </PixelCard>

        {/* Keyboard Visualization */}
        <PixelCard variant="default" className="mb-8">
          <h3 className="font-pixel text-sm mb-4 text-center">Keyboard (60% Layout)</h3>
          <div className="flex flex-col items-center gap-2">
            {keyboardLayout.map((row, rowIdx) => (
              <div key={rowIdx} className="flex gap-2 justify-center">
                {row.map((key) => (
                  <div
                    key={key}
                    className="pixel-border w-12 h-12 flex items-center justify-center font-pixel text-sm bg-muted"
                  >
                    {key}
                  </div>
                ))}
              </div>
            ))}
            <div className="pixel-border w-[400px] h-12 flex items-center justify-center font-pixel text-sm bg-muted mt-2">
              SPACE
            </div>
          </div>
        </PixelCard>

        <div className="flex justify-center">
          <PixelButton variant="primary" size="lg" onClick={() => navigate("/student/play")}>
            Finish Session
          </PixelButton>
        </div>
      </div>
    </div>
  );
}
