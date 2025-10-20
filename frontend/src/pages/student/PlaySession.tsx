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

  const [words, setWords] = useState<string[]>([]);
  const [typedWords, setTypedWords] = useState<string[]>([]);
  const [currentWordIndex, setCurrentWordIndex] = useState(0);
  const [userInput, setUserInput] = useState("");
  const [activeKeys, setActiveKeys] = useState<{ [key: string]: string }>({}); // 🔹 Tracks key highlight colors
  const [startTime] = useState(Date.now());
  const [wpm, setWpm] = useState(0);
  const [accuracy, setAccuracy] = useState(100);

  const keyboardLayout = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
    ["Z", "X", "C", "V", "B", "N", "M"],
  ];

  // 🔹 Fetch words
  useEffect(() => {
    const fetchTypingData = async () => {
      try {
        const res = await fetch(`http://localhost:5000/api/typing/level/${level}`);
        const data = await res.json();
        if (data && data.data) {
          const text = data.data.join(" ");
          setWords(text.split(" "));
        }
      } catch (error) {
        console.error("Error fetching typing data:", error);
      }
    };
    fetchTypingData();
  }, [level]);

  // 🔹 Handle typing
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;

    // Space → submit current word
    if (value.endsWith(" ")) {
      const typedWord = value.trim();

      setTypedWords((prev) => {
        const updated = [...prev];
        updated[currentWordIndex] = typedWord;
        return updated;
      });

      setCurrentWordIndex((prev) => prev + 1);
      setUserInput("");
      return;
    }

    setUserInput(value);
  };

  // 🔹 Prevent backspace/delete & record key color feedback
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Backspace" || e.key === "Delete") {
      e.preventDefault();
      return;
    }

    const pressedKey = e.key.toUpperCase();
    if (!/^[A-Z]$/i.test(pressedKey)) return; // only letters light up

    const currentWord = words[currentWordIndex] || "";
    const typedIndex = userInput.length;
    const correctChar = currentWord[typedIndex]?.toUpperCase();

    const color =
      pressedKey === correctChar ? "bg-green-500 text-white" : "bg-red-500 text-white";

    setActiveKeys((prev) => ({
      ...prev,
      [pressedKey]: color,
    }));

    // fade key after 300ms
    setTimeout(() => {
      setActiveKeys((prev) => {
        const updated = { ...prev };
        delete updated[pressedKey];
        return updated;
      });
    }, 300);
  };

  // 🔹 Compute WPM & Accuracy
  useEffect(() => {
    const correctCount = typedWords.filter(
      (typed, i) => typed && typed === words[i]
    ).length;
    const totalTyped = typedWords.filter(Boolean).length;
    const accuracyVal = totalTyped > 0 ? Math.round((correctCount / totalTyped) * 100) : 100;

    const timeElapsed = (Date.now() - startTime) / 1000 / 60;
    setWpm(Math.round(correctCount / timeElapsed) || 0);
    setAccuracy(accuracyVal);
  }, [typedWords, startTime]);

  // 🔹 Per-letter rendering
  const renderWord = (word: string, index: number) => {
    const typed = index === currentWordIndex ? userInput : typedWords[index] || "";
    const isCurrent = index === currentWordIndex;

    return (
      <span key={index} className="mr-3">
        {word.split("").map((char, i) => {
          let color = "text-muted-foreground";

          if (i < typed.length) {
            color = typed[i] === char ? "text-green-600" : "text-red-600";
          } else if (isCurrent && i === typed.length) {
            color = "text-orange-400 underline animate-pulse";
          }

          return (
            <span key={i} className={`font-pixel ${color}`}>
              {char}
            </span>
          );
        })}
        {typed.length > word.length && (
          <span className="font-pixel text-red-600">{typed.slice(word.length)}</span>
        )}
      </span>
    );
  };

  const handleFinish = () => navigate("/student/play");
  const isFinished = currentWordIndex >= words.length;

  return (
    <div className="min-h-screen p-8 flex flex-col items-center">
      <div className="max-w-7xl w-full">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <PixelButton variant="secondary" onClick={() => navigate("/student/play")}>
              <ArrowLeft size={20} />
            </PixelButton>
            <Logo />
          </div>
          <div className="font-pixel text-sm">
            {sessionType === "evaluated" ? "🏆 Graded Session" : "🎮 Practice Mode"} - Level{" "}
            {level}
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <PixelCard variant="yellow">
            <p className="font-pixel text-xs mb-1">WPM</p>
            <p className="font-pixel text-2xl">{wpm}</p>
          </PixelCard>
          <PixelCard variant="orange">
            <p className="font-pixel text-xs mb-1">Accuracy</p>
            <p className="font-pixel text-2xl">{accuracy}%</p>
          </PixelCard>
          <PixelCard variant="green">
            <p className="font-pixel text-xs mb-1">Completed</p>
            <p className="font-pixel text-2xl">{typedWords.filter(Boolean).length}</p>
          </PixelCard>
          <PixelCard variant="red">
            <p className="font-pixel text-xs mb-1">Remaining</p>
            <p className="font-pixel text-2xl">
              {words.length - typedWords.filter(Boolean).length}
            </p>
          </PixelCard>
        </div>

        {/* Typing area */}
        {!isFinished ? (
          <PixelCard className="mb-8 flex flex-col items-center justify-center text-center py-8">
            {/* Words */}
            <div className="font-pixel text-lg mb-6 flex flex-wrap justify-center gap-2 max-w-3xl leading-relaxed">
              {words.map((word, index) => renderWord(word, index))}
            </div>

            {/* Input */}
            <div className="flex justify-center w-full mb-8">
              <input
                type="text"
                value={userInput}
                onChange={handleChange}
                onKeyDown={handleKeyDown}
                className="text-center w-2/3 md:w-1/2 px-4 py-3 bg-input border-[3px] border-border text-foreground font-pixel text-lg focus:outline-none focus:ring-2 focus:ring-primary rounded-md"
                placeholder="Type here..."
                autoFocus
              />
            </div>

            {/* Keyboard Layout */}
            <div className="font-pixel text-sm text-muted-foreground mb-4">
              Keyboard (60% Layout)
            </div>
            <div className="flex flex-col items-center gap-2">
              {keyboardLayout.map((row, rowIdx) => (
                <div key={rowIdx} className="flex gap-2 justify-center">
                  {row.map((key) => (
                    <div
                      key={key}
                      className={`pixel-border w-10 h-10 flex items-center justify-center font-pixel text-sm border border-border rounded-md transition-colors duration-150 ${
                        activeKeys[key]
                          ? activeKeys[key]
                          : "bg-muted text-foreground"
                      }`}
                    >
                      {key}
                    </div>
                  ))}
                </div>
              ))}
              <div
                className={`pixel-border w-[300px] h-10 flex items-center justify-center font-pixel text-sm border border-border rounded-md mt-2 ${
                  activeKeys[" "] ? activeKeys[" "] : "bg-muted text-foreground"
                }`}
              >
                SPACE
              </div>
            </div>
          </PixelCard>
        ) : (
          <PixelCard className="text-center py-8">
            <p className="font-pixel text-xl text-green-600 mb-4">🎉 Session Complete!</p>
            <p className="font-pixel text-sm text-muted-foreground">
              You typed {typedWords.length} words with {accuracy}% accuracy.
            </p>
          </PixelCard>
        )}

        {/* Finish Button */}
        <div className="flex justify-center mt-6">
          <PixelButton variant="primary" size="lg" onClick={handleFinish}>
            {isFinished ? "Back to Play" : "Finish Session"}
          </PixelButton>
        </div>
      </div>
    </div>
  );
}