import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Logo } from "@/components/Logo";
import { PixelButton } from "@/components/PixelButton";
import { PixelCard } from "@/components/PixelCard";
import { ArrowLeft } from "lucide-react";
import bgImage from "@/assets/b1.jpg";

export default function HomeRowModule() {
  const navigate = useNavigate();

  const lessons = [
    "a s d f j k l ;",
    "asdf jkl; asdf jkl;",
    "f j f j f j f j",
    "asdfg hjkl; asdfg hjkl;",
    "aa ss dd ff jj kk ll ;;",
    "asdf fjkl asdf fjkl;",
    "sad lad fad dad jak kal laf; dak;",
    "fall salad flask ask jaff kal fad;",
    "as dj fk la sj kd lf aj sk dl fj ka;",
    "a sad lad falls as jaffa asks falkal;",
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

  const homeRowLayout = [["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"]];

  // WPM + Accuracy
  useEffect(() => {
    const elapsed = (Date.now() - lessonStart) / 1000 / 60;
    const wordsTyped = userInput.trim().split(" ").length;
    setWpm(Math.round(wordsTyped / (elapsed || 1)) || 0);
    let correct = 0;
    for (let i = 0; i < userInput.length; i++)
      if (userInput[i] === currentText[i]) correct++;
    setAccuracy(
      userInput.length > 0 ? Math.round((correct / userInput.length) * 100) : 100
    );
  }, [userInput, lessonStart, currentText]);

  // Auto next
  useEffect(() => {
    if (userInput.length >= currentText.length && !completed) {
      const t = setTimeout(() => handleNext(), 500);
      return () => clearTimeout(t);
    }
  }, [userInput, currentText]);

  const handleTyping = (val) => {
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
    const newErrors = { ...errorMap };
    for (let i = 0; i < userInput.length; i++) {
      const expected = currentText[i];
      const typed = userInput[i];
      if (typed === expected) correctCount++;
      else if (expected && expected !== " ")
        newErrors[expected] = (newErrors[expected] || 0) + 1;
    }
    setTotalChars((p) => p + userInput.length);
    setTotalCorrect((p) => p + correctCount);
    setTotalTime((p) => p + elapsed);
    setErrorMap(newErrors);

    if (currentIndex < lessons.length - 1) {
      setCurrentIndex(currentIndex + 1);
      setCurrentText(lessons[currentIndex + 1]);
      setUserInput("");
      setLessonStart(Date.now());
    } else {
      const finalElapsed = (Date.now() - lessonStart) / 1000;
      const finalTime = totalTime + finalElapsed;
      const finalChars = totalChars + userInput.length;
      const finalCorrect = totalCorrect + correctCount;
      const gross = (finalChars / 5) / (finalTime / 60);
      const acc =
        finalChars > 0 ? Math.round((finalCorrect / finalChars) * 100) : 100;
      setWpm(Math.round(gross));
      setAccuracy(acc);
      setCompleted(true);
    }
  };

  const handleFinish = () => {
    const saved = JSON.parse(localStorage.getItem("typingModuleProgress")) || {};
    saved[1] = { completed: true };
    localStorage.setItem("typingModuleProgress", JSON.stringify(saved));
    navigate("/student/learn");
  };

  const getPerformanceMessage = () =>
    wpm >= 40 && accuracy >= 95
      ? "Excellent work! You’re typing like a pro!"
      : wpm >= 25 && accuracy >= 85
      ? "Great job! You’re improving quickly."
      : "Keep practicing! Focus on accuracy before speed.";

  const getWeakKeys = () => {
    const entries = Object.entries(errorMap);
    if (entries.length === 0)
      return "You made no significant key errors. Great job!";
    const sorted = [...entries].sort((a, b) => b[1] - a[1]);
    const top = sorted.slice(0, 3).map(([k]) => k.toUpperCase());
    return `You had the most trouble with: ${top.join(", ")}`;
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
              Module 1: Home Row Keys
            </h1>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            <PixelCard variant="yellow"><p className="font-pixel text-xs mb-1">Lesson</p><p className="font-pixel text-2xl">{currentIndex + 1}/10</p></PixelCard>
            <PixelCard variant="orange"><p className="font-pixel text-xs mb-1">WPM</p><p className="font-pixel text-2xl">{wpm}</p></PixelCard>
            <PixelCard variant="purple"><p className="font-pixel text-xs mb-1">Accuracy</p><p className="font-pixel text-2xl">{accuracy}%</p></PixelCard>
            <PixelCard variant="green"><p className="font-pixel text-xs mb-1">Progress</p><p className="font-pixel text-2xl">{Math.round(((currentIndex + 1) / lessons.length) * 100)}%</p></PixelCard>
          </div>

          {/* Typing area */}
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
                  : `Lesson ${currentIndex + 1}: Type using ASDF JKL;`
              }
              autoFocus
            />
          </PixelCard>

          {/* ✅ Restored Keyboard Layout */}
          <PixelCard className="mb-8 text-center bg-black/60 border-[3px] border-yellow-300 backdrop-blur-sm">
            <h3 className="font-pixel text-sm mb-4 text-white">Home Row Layout</h3>
            <div className="flex justify-center gap-2">
              {homeRowLayout[0].map((key) => {
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

          {/* Completion */}
          {completed && (
            <div className="text-center">
              <PixelCard className="inline-block p-8 bg-black/70 border-[3px] border-yellow-300 backdrop-blur-md">
                <h2 className="font-pixel text-2xl mb-4 text-yellow-200">Module Complete!</h2>
                <p className="font-pixel mb-4">{getPerformanceMessage()}</p>
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
