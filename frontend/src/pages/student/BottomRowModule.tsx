import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Logo } from "@/components/Logo";
import { PixelButton } from "@/components/PixelButton";
import { PixelCard } from "@/components/PixelCard";
import { ArrowLeft } from "lucide-react";
import bgImage from "@/assets/b3.jpg";

export default function BottomRowModule() {
  const navigate = useNavigate();

  // 🔹 Bottom-row typing drills
  const lessons = [
    "z x c v n m , .",
    "zxcv nm,. zxcv nm,.",
    "z n z n z n z n",
    "zxcvbnm,. zxcvbnm,.",
    "zz xx cc vv nn mm ,, ..",
    "zxcv nm,. zxcv nm,.",
    "zany mix calm move zoom",
    "vex man can jam zen",
    "z x c v n m , .",
    "zen man can move calm",
  ];

  const [currentIndex, setCurrentIndex] = useState(0);
  const [currentText, setCurrentText] = useState(lessons[0]);
  const [userInput, setUserInput] = useState("");
  const [wpm, setWpm] = useState(0);
  const [accuracy, setAccuracy] = useState(100);
  const [completed, setCompleted] = useState(false);
  const [lessonStart, setLessonStart] = useState(Date.now());
  const [highlightedKey, setHighlightedKey] = useState("");
  const [totalChars, setTotalChars] = useState(0);
  const [totalCorrect, setTotalCorrect] = useState(0);
  const [totalTime, setTotalTime] = useState(0);
  const [errorMap, setErrorMap] = useState<Record<string, number>>({});

  const bottomRowLayout = [["Z", "X", "C", "V", "B", "N", "M", ",", "."]];

  // 🔹 WPM + accuracy calculation
  useEffect(() => {
    const elapsed = (Date.now() - lessonStart) / 1000 / 60;
    const words = userInput.trim().split(" ").length;
    setWpm(Math.round(words / (elapsed || 1)) || 0);

    let correct = 0;
    for (let i = 0; i < userInput.length; i++)
      if (userInput[i] === currentText[i]) correct++;
    setAccuracy(
      userInput.length ? Math.round((correct / userInput.length) * 100) : 100
    );
  }, [userInput, lessonStart, currentText]);

  // 🔹 Move to next lesson when finished
  useEffect(() => {
    if (userInput.length >= currentText.length && !completed) {
      const t = setTimeout(() => handleNext(), 500);
      return () => clearTimeout(t);
    }
  }, [userInput, currentText]);

  const handleTyping = (val: string) => {
    if (completed) return;
    setUserInput(val);
    const last = val[val.length - 1];
    if (last) {
      setHighlightedKey(last.toUpperCase());
      setTimeout(() => setHighlightedKey(""), 300);
    }
  };

  const handleNext = () => {
    const elapsed = (Date.now() - lessonStart) / 1000;
    let correctCount = 0;
    const newErr: Record<string, number> = { ...errorMap };

    for (let i = 0; i < userInput.length; i++) {
      const exp = currentText[i];
      const t = userInput[i];
      if (t === exp) correctCount++;
      else if (exp && exp !== " ") newErr[exp] = (newErr[exp] || 0) + 1;
    }

    setTotalChars((p) => p + userInput.length);
    setTotalCorrect((p) => p + correctCount);
    setTotalTime((p) => p + elapsed);
    setErrorMap(newErr);

    if (currentIndex < lessons.length - 1) {
      setCurrentIndex(currentIndex + 1);
      setCurrentText(lessons[currentIndex + 1]);
      setUserInput("");
      setLessonStart(Date.now());
    } else {
      const finE = (Date.now() - lessonStart) / 1000;
      const finT = totalTime + finE;
      const finC = totalChars + userInput.length;
      const finCor = totalCorrect + correctCount;
      const gross = (finC / 5) / (finT / 60);
      const acc = finC ? Math.round((finCor / finC) * 100) : 100;
      setWpm(Math.round(gross));
      setAccuracy(acc);
      setCompleted(true);
    }
  };

  const handleFinish = () => {
    const saved = JSON.parse(localStorage.getItem("typingModuleProgress") || "{}");
    saved[3] = { completed: true };
    localStorage.setItem("typingModuleProgress", JSON.stringify(saved));
    navigate("/student/learn");
  };

  const getPerformanceMessage = () =>
    wpm >= 40 && accuracy >= 95
      ? "Excellent control! You’ve conquered the bottom row!"
      : wpm >= 25 && accuracy >= 85
      ? "Nice job! Keep working on your finger precision."
      : "Good work! Keep practicing for smoother transitions.";

  const getWeakKeys = () => {
    const ent = Object.entries(errorMap);
    if (ent.length === 0) return "No major mistakes detected!";
    const sorted = [...ent].sort((a, b) => b[1] - a[1]);
    const top = sorted.slice(0, 3).map(([k]) => k.toUpperCase());
    return `You need more practice with: ${top.join(", ")}`;
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
          {/* Header */}
          <div className="flex items-center justify-between mb-8">
            <div className="flex items-center gap-4">
              <PixelButton variant="secondary" onClick={() => navigate("/student/learn")}>
                <ArrowLeft size={20} />
              </PixelButton>
              <Logo />
            </div>
            <h1 className="font-pixel text-xl text-black">
              Module 3: Bottom Row Keys
            </h1>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            <PixelCard variant="yellow"><p className="font-pixel text-xs mb-1">Lesson</p><p className="font-pixel text-2xl">{currentIndex + 1}/10</p></PixelCard>
            <PixelCard variant="orange"><p className="font-pixel text-xs mb-1">WPM</p><p className="font-pixel text-2xl">{wpm}</p></PixelCard>
            <PixelCard variant="purple"><p className="font-pixel text-xs mb-1">Accuracy</p><p className="font-pixel text-2xl">{accuracy}%</p></PixelCard>
            <PixelCard variant="green"><p className="font-pixel text-xs mb-1">Progress</p><p className="font-pixel text-2xl">{Math.round(((currentIndex + 1) / lessons.length) * 100)}%</p></PixelCard>
          </div>

          {/* Typing Area */}
          <PixelCard className="mb-8 bg-black/60 border-[3px] border-yellow-300 backdrop-blur-sm">
            <div className="font-pixel text-lg mb-4 text-center">
              {currentText.split("").map((ch, i) => {
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
              disabled={completed}
              className={`w-full px-4 py-3 font-pixel text-sm border-[3px] ${
                completed
                  ? "bg-gray-700 border-gray-500 text-gray-400"
                  : "bg-black/70 border-yellow-300 text-white"
              }`}
              placeholder={
                completed
                  ? "Module completed!"
                  : `Lesson ${currentIndex + 1}: Type using ZXCV NM,.`
              }
              autoFocus
            />
          </PixelCard>

          {/* 🔹 Keyboard Layout */}
          <PixelCard className="mb-8 text-center bg-black/60 border-[3px] border-yellow-300 backdrop-blur-sm">
            <h3 className="font-pixel text-sm mb-4 text-white">Bottom Row Layout</h3>
            <div className="flex justify-center gap-2">
              {bottomRowLayout[0].map((key) => {
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

          {/* Completion Section */}
          {completed && (
            <div className="text-center">
              <PixelCard className="inline-block p-8 bg-black/70 border-[3px] border-yellow-300 backdrop-blur-md">
                <h2 className="font-pixel text-2xl mb-4 text-yellow-200">
                  Module Complete!
                </h2>
                <p className="font-pixel mb-4">{getPerformanceMessage()}</p>
                <p className="font-pixel mb-6 text-gray-300">{getWeakKeys()}</p>
                <div className="grid grid-cols-2 gap-4 mb-6">
                  <PixelCard className="bg-black/50 border border-yellow-200 text-center"><p className="font-pixel text-xs mb-1 text-yellow-200">WPM</p><p className="font-pixel text-2xl">{wpm}</p></PixelCard>
                  <PixelCard className="bg-black/50 border border-yellow-200 text-center"><p className="font-pixel text-xs mb-1 text-yellow-200">Accuracy</p><p className="font-pixel text-2xl">{accuracy}%</p></PixelCard>
                </div>
                <PixelButton variant="primary" size="lg" onClick={handleFinish}>
                  Return to Modules
                </PixelButton>
              </PixelCard>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
