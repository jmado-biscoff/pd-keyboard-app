import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Logo } from "@/components/Logo";
import { PixelButton } from "@/components/PixelButton";
import { PixelCard } from "@/components/PixelCard";
import { ArrowLeft } from "lucide-react";
import bgImage from "@/assets/b6.jpg";

export default function FullKeyboardModule() {
  const navigate = useNavigate();

  // 🔹 Pool of ALL lessons (technical + paragraph)
  const allLessons = [
    // --- Technical keyboard drills ---
    "a s d f j k l ; q w e r u i o p z x c v n m , . 1 2 3 4 5 6 7 8 9 0",
    "qaz wsx edc rfv tgb yhn ujm ik, ol. p;/ 1234567890",
    "asdf jkl; qwer uiop zxcv nm,. 0987654321",
    "! @ # $ % ^ & * ( ) q w e r t y u i o p",
    "a1 s2 d3 f4 j5 k6 l7 ;8 q9 w0 e1 r2 t3 y4",
    "z x c v b n m , . ! @ # $ % ^ & * ( )",
    "1q 2w 3e 4r 5t 6y 7u 8i 9o 0p",

    // --- Paragraph practice drills ---
    "Typing is a skill that improves with daily practice and patience. Keep your posture straight and your fingers light on the keys.",
    "Start slow and focus on accuracy. Every correct keystroke builds muscle memory and confidence over time.",
    "The home row keys act as your guide. Rest your fingers there and reach out only when necessary.",
    "Avoid looking at the keyboard. Your brain learns patterns faster when you rely on touch instead of sight.",
    "Take short breaks between sessions to prevent fatigue. A relaxed mind types more smoothly and accurately.",
    "Good lighting and a proper chair can make typing for long periods more comfortable and enjoyable.",
    "Typing paragraphs like this helps train rhythm, spacing, and punctuation awareness naturally.",
    "Consistency matters more than speed. A few minutes of mindful typing each day brings better results than long but irregular practice.",
    "Numbers are part of daily typing too. Try entering 12345 smoothly while keeping your focus on form.",
    "When you make a mistake, correct it calmly. The goal is improvement, not perfection in one session.",
    "Typing short stories or diary entries is a fun way to apply everything you’ve learned so far.",
    "Accuracy first, then speed. Once you type without errors, your speed will naturally follow.",
    "Remember to use both hands equally. Each finger has its role, and balance makes your movements efficient.",
    "With steady progress, you’ll soon type effortlessly across letters, numbers, and punctuation.",
    "Congratulations! You’ve reached the final stage of your typing journey. Keep practicing daily to maintain mastery.",
  ];

  // 🔹 Randomly pick 10 unique lessons
  const shuffled = [...allLessons].sort(() => Math.random() - 0.5).slice(0, 10);

  const [lessons] = useState(shuffled);
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

  // 🔹 Layout visualization
  const fullLayout = [
    ["1","2","3","4","5","6","7","8","9","0"],
    ["Q","W","E","R","T","Y","U","I","O","P"],
    ["A","S","D","F","G","H","J","K","L",";"],
    ["Z","X","C","V","B","N","M",",","."],
  ];

  // 🔹 Compute WPM & Accuracy
  useEffect(() => {
    const elapsed = (Date.now() - lessonStart) / 1000 / 60;
    const words = userInput.trim().split(/\s+/).length;
    setWpm(Math.round(words / (elapsed || 1)) || 0);

    let correct = 0;
    for (let i = 0; i < userInput.length; i++)
      if (userInput[i] === currentText[i]) correct++;
    setAccuracy(
      userInput.length ? Math.round((correct / userInput.length) * 100) : 100
    );
  }, [userInput, lessonStart, currentText]);

  // 🔹 Auto-next lesson
  useEffect(() => {
    if (userInput.length >= currentText.length && !completed) {
      const t = setTimeout(() => handleNext(), 700);
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
    const newErr = { ...errorMap };

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
    saved[6] = { completed: true };
    localStorage.setItem("typingModuleProgress", JSON.stringify(saved));
    navigate("/student/learn");
  };

  const getPerformanceMessage = () =>
    wpm >= 45 && accuracy >= 95
      ? "Excellent! You’ve achieved full keyboard mastery!"
      : wpm >= 30 && accuracy >= 85
      ? "Great progress! You type fluently and accurately across all keys."
      : "Good work! Keep practicing to perfect your rhythm and accuracy.";

  const getWeakKeys = () => {
    const ent = Object.entries(errorMap);
    if (ent.length === 0) return "No major mistakes detected!";
    const sorted = [...ent].sort((a, b) => b[1] - a[1]);
    const top = sorted.slice(0, 5).map(([k]) => k.toUpperCase());
    return `Keys to review: ${top.join(", ")}`;
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
            <h1 className="font-pixel text-xl text-white">
              Module 6: Full Keyboard
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
            <div className="font-pixel text-base mb-4 text-center leading-relaxed px-4">
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
                  : `Lesson ${currentIndex + 1}: Type the text above`
              }
              autoFocus
            />
          </PixelCard>

          {/* Keyboard Layout */}
          <PixelCard className="mb-8 text-center bg-black/60 border-[3px] border-yellow-300 backdrop-blur-sm">
            <h3 className="font-pixel text-sm mb-4 text-white">Keyboard Layout</h3>
            <div className="flex flex-col items-center gap-2">
              {fullLayout.map((row, idx) => (
                <div key={idx} className="flex justify-center gap-1">
                  {row.map((key) => {
                    const active = key === highlightedKey;
                    return (
                      <div
                        key={key}
                        className={`pixel-border w-9 h-9 flex items-center justify-center font-pixel text-xs border-[2px] transition-all duration-150 ${
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
              ))}
            </div>
          </PixelCard>

          {/* Completion */}
          {completed && (
            <div className="text-center">
              <PixelCard className="inline-block p-8 bg-black/70 border-[3px] border-yellow-300 backdrop-blur-md">
                <h2 className="font-pixel text-2xl mb-4 text-yellow-200">
                  Module Complete!
                </h2>
                <p className="font-pixel mb-4">{getPerformanceMessage()}</p>
                <p className="font-pixel mb-6 text-gray-300">{getWeakKeys()}</p>
                <div className="grid grid-cols-2 gap-4 mb-6">
                  <PixelCard className="bg-black/50 border border-yellow-200 text-center">
                    <p className="font-pixel text-xs mb-1 text-yellow-200">WPM</p>
                    <p className="font-pixel text-2xl">{wpm}</p>
                  </PixelCard>
                  <PixelCard className="bg-black/50 border border-yellow-200 text-center">
                    <p className="font-pixel text-xs mb-1 text-yellow-200">Accuracy</p>
                    <p className="font-pixel text-2xl">{accuracy}%</p>
                  </PixelCard>
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
