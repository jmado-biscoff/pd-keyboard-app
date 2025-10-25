import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Logo } from "@/components/Logo";
import { PixelButton } from "@/components/PixelButton";
import { PixelCard } from "@/components/PixelCard";
import { ArrowLeft } from "lucide-react";
import bgImage from "@/assets/b2.jpg";

export default function TopRowModule() {
  const navigate = useNavigate();

  const lessons = [
    "q w e r u i o p",
    "qwer uiop qwer uiop",
    "r e w q p o i u",
    "we rq op iu we rq op iu",
    "qq ww ee rr uu ii oo pp",
    "qwer uiop qwer uiop",
    "weir pour ripe wire pure",
    "quire poet rope wire pour",
    "q w e r i o p u q w e r",
    "we rip our wire pure quip rope",
  ];

  const [currentIndex, setCurrentIndex] = useState(0);
  const [userInput, setUserInput] = useState("");
  const [wpm, setWpm] = useState(0);
  const [accuracy, setAccuracy] = useState(100);
  const [completed, setCompleted] = useState(false);
  const [highlightedKey, setHighlightedKey] = useState("");
  const [lessonStart, setLessonStart] = useState(Date.now());
  const [elapsedTime, setElapsedTime] = useState(0);
  const [errorMap, setErrorMap] = useState<Record<string, number>>({});
  const [timerRunning, setTimerRunning] = useState(true);

  const topRowLayout = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"]];

  useEffect(() => {
    let interval: any;
    if (timerRunning && !completed) {
      interval = setInterval(() => {
        setElapsedTime(Math.floor((Date.now() - lessonStart) / 1000));
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [timerRunning, completed, lessonStart]);

  const currentText = lessons[currentIndex];

  useEffect(() => {
    const elapsedMinutes = Math.max((Date.now() - lessonStart) / 1000 / 60, 0.01);
    const wordsTyped = userInput.trim().split(" ").length;
    setWpm(Math.round(wordsTyped / elapsedMinutes));

    let correct = 0;
    for (let i = 0; i < userInput.length; i++) {
      if (userInput[i] === currentText[i]) correct++;
    }
    setAccuracy(userInput.length > 0 ? Math.round((correct / userInput.length) * 100) : 100);
  }, [userInput, lessonStart, currentText]);

  useEffect(() => {
    if (userInput.length >= currentText.length && !completed) {
      setTimeout(() => handleNext(), 400);
    }
  }, [userInput, currentText]);

  const handleTyping = (val: string) => {
    if (completed) return;
    setUserInput(val);
    const last = val[val.length - 1];
    if (last) {
      setHighlightedKey(last.toUpperCase());
      setTimeout(() => setHighlightedKey(""), 200);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Backspace" || e.key === "Delete") e.preventDefault();
  };

  const handleNext = () => {
    const newErrors = { ...errorMap };
    for (let i = 0; i < userInput.length; i++) {
      const expected = currentText[i];
      const typed = userInput[i];
      if (typed !== expected && expected && expected !== " ")
        newErrors[expected] = (newErrors[expected] || 0) + 1;
    }

    setErrorMap(newErrors);
    if (currentIndex < lessons.length - 1) {
      setCurrentIndex((prev) => prev + 1);
      setUserInput("");
      setLessonStart(Date.now());
      setElapsedTime(0);
    } else {
      setCompleted(true);
      setTimerRunning(false);
    }
  };

  const handleFinish = () => {
    const saved = JSON.parse(localStorage.getItem("typingModuleProgress") || "{}");
    saved[2] = { completed: true };
    localStorage.setItem("typingModuleProgress", JSON.stringify(saved));
    navigate("/student/learn");
  };

  const fingerMap: Record<string, string> = {
    Q: "Left Pinky",
    W: "Left Ring",
    E: "Left Middle",
    R: "Left Index",
    U: "Right Index",
    I: "Right Middle",
    O: "Right Ring",
    P: "Right Pinky",
  };

  const getWeakKeys = () => {
    const entries = Object.entries(errorMap);
    if (entries.length === 0) return "No key mistakes! Great job!";
    const sorted = entries.sort((a, b) => b[1] - a[1]);
    const top = sorted.slice(0, 3).map(([k]) => k.toUpperCase());
    const feedback = top.map((k) => `${k} → ${fingerMap[k] || "Unknown Finger"}`).join(", ");
    return `You struggled with: ${feedback}`;
  };

  return (
    <div
      className="relative min-h-screen overflow-hidden"
      style={{
        backgroundImage: `url(${bgImage})`,
        backgroundSize: "cover",
        backgroundPosition: "center",
      }}
    >
      <div className="absolute inset-0 bg-black/30 z-0" />
      <div className="relative z-10 p-8 min-h-screen text-white">
        <div className="max-w-6xl mx-auto">
          <div className="flex items-center justify-between mb-8">
            <div className="flex items-center gap-4">
              <PixelButton variant="secondary" onClick={() => navigate("/student/learn")}>
                <ArrowLeft size={20} />
              </PixelButton>
              <Logo />
            </div>
            <h1 className="font-pixel text-xl text-black">Module 2: Top Row Keys</h1>
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-5 gap-4 mb-8">
            <PixelCard variant="yellow"><p className="font-pixel text-xs mb-1">Lesson</p><p className="font-pixel text-2xl">{currentIndex + 1}/10</p></PixelCard>
            <PixelCard variant="orange"><p className="font-pixel text-xs mb-1">WPM</p><p className="font-pixel text-2xl">{wpm}</p></PixelCard>
            <PixelCard variant="purple"><p className="font-pixel text-xs mb-1">Accuracy</p><p className="font-pixel text-2xl">{accuracy}%</p></PixelCard>
            <PixelCard variant="green"><p className="font-pixel text-xs mb-1">Timer</p><p className="font-pixel text-2xl">{elapsedTime}s</p></PixelCard>
            <PixelCard variant="green"><p className="font-pixel text-xs mb-1">Progress</p><p className="font-pixel text-2xl">{Math.round(((currentIndex + 1) / lessons.length) * 100)}%</p></PixelCard>
          </div>

          <PixelCard className="mb-8 bg-black/60 border-[3px] border-yellow-300 backdrop-blur-sm">
            <div className="font-pixel text-lg mb-4 text-center">
              {lessons[currentIndex].split("").map((ch, i) => {
                let c = "text-gray-400";
                if (i < userInput.length)
                  c = userInput[i] === ch ? "text-green-400" : "text-red-500";
                else if (i === userInput.length && !completed)
                  c = "text-purple-400 animate-pulse";
                return <span key={i} className={c}>{ch}</span>;
              })}
            </div>
            <input
              type="text"
              value={userInput}
              onChange={(e) => handleTyping(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={completed}
              className={`w-full px-4 py-3 font-pixel text-sm border-[3px] ${
                completed
                  ? "bg-gray-700 border-gray-500 text-gray-400"
                  : "bg-black/70 border-yellow-300 text-white"
              }`}
              placeholder={completed ? "Module completed!" : "Type the keys above..."}
              autoFocus
            />
          </PixelCard>

          <PixelCard className="mb-8 text-center bg-black/60 border-[3px] border-yellow-300 backdrop-blur-sm">
            <h3 className="font-pixel text-sm mb-4 text-white">Top Row Layout</h3>
            <div className="flex justify-center gap-2">
              {topRowLayout[0].map((key) => {
                const active = key === highlightedKey;
                return (
                  <div
                    key={key}
                    className={`pixel-border w-12 h-12 flex items-center justify-center font-pixel text-sm border-[2px] transition-all duration-150 ${
                      active
                        ? "bg-purple-500 text-white scale-110 shadow-[0_0_10px_#a855f7]"
                        : "bg-gray-800 text-yellow-100 border-yellow-400"
                    }`}
                  >
                    {key}
                  </div>
                );
              })}
            </div>
          </PixelCard>

          {completed && (
            <div className="text-center">
              <PixelCard className="inline-block p-8 bg-black/70 border-[3px] border-yellow-300 backdrop-blur-md">
                <h2 className="font-pixel text-2xl mb-4 text-yellow-200">Module Complete!</h2>
                <p className="font-pixel mb-4">
                  {accuracy >= 95
                    ? "Excellent work!"
                    : accuracy >= 85
                    ? "Nice job! Keep improving!"
                    : "Focus on accuracy first."}
                </p>
                <p className="font-pixel mb-6 text-gray-300">{getWeakKeys()}</p>
                <div className="grid grid-cols-2 gap-4 mb-6">
                  <PixelCard className="bg-black/50 border border-yellow-200 text-center"><p className="font-pixel text-xs mb-1 text-yellow-200">WPM</p><p className="font-pixel text-2xl">{wpm}</p></PixelCard>
                  <PixelCard className="bg-black/50 border border-yellow-200 text-center"><p className="font-pixel text-xs mb-1 text-yellow-200">Accuracy</p><p className="font-pixel text-2xl">{accuracy}%</p></PixelCard>
                </div>
                <PixelButton variant="primary" size="lg" onClick={handleFinish}>Return to Modules</PixelButton>
              </PixelCard>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
