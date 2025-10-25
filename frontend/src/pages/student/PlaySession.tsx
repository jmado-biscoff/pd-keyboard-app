import { useState, useEffect, useRef } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { Logo } from "@/components/Logo";
import { PixelButton } from "@/components/PixelButton";
import { PixelCard } from "@/components/PixelCard";
import { ArrowLeft } from "lucide-react";

const BASE_URL = import.meta.env.VITE_API_URL.replace("/api/auth", "");

// ============================================================
// Session Report Interface (Phase 1 – Metrics Foundation)
// ============================================================
interface SessionReport {
  wpm: number;
  accuracy: number;
  correct_keystrokes: number;
  incorrect_keystrokes: number;
  fingerAccuracy: number;       // % of correct finger-to-key matches
  timingVariance: number;       // key interval variance in ms
  session_duration_sec: number; // total session time
}

async function startDetection() {
  const res = await fetch(`${BASE_URL}/api/detect/start`, { method: "POST" });
  return res.json();
}
async function stopDetection() {
  const res = await fetch(`${BASE_URL}/api/detect/stop`, { method: "POST" });
  return res.json();
}
async function getDetectionStatus() {
  const res = await fetch(`${BASE_URL}/api/detect/status`);
  return res.json();
}

export default function PlaySession() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const sessionType = searchParams.get("type") || "practice";
  const level = parseInt(searchParams.get("level") || "1");

  const [words, setWords] = useState<string[]>([]);
  const [typedWords, setTypedWords] = useState<string[]>([]);
  const [currentWordIndex, setCurrentWordIndex] = useState(0);
  const [userInput, setUserInput] = useState("");
  const [activeKeys, setActiveKeys] = useState<{ [key: string]: string }>({});
  const [charFeedback, setCharFeedback] = useState<{ [index: number]: string }>({});
  const [startTime] = useState(Date.now());
  const [wpm, setWpm] = useState(0);
  const [accuracy, setAccuracy] = useState(100);
  const [detecting, setDetecting] = useState(false);
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [calibrationDone, setCalibrationDone] = useState(false);
  const [correctCount, setCorrectCount] = useState(0);
  const [incorrectCount, setIncorrectCount] = useState(0);
  const [lastKey, setLastKey] = useState<string | null>(null);

  // ✅ NEW: Track last detection signature to prevent duplicate counting
  const lastEventRef = useRef<string | null>(null);

  // ✅ Additional metric states (Phase 1)
  const [fingerAccuracy, setFingerAccuracy] = useState(0);
  const [timingVariance, setTimingVariance] = useState(0);

  // 🧠 Track individual key errors
  const [errorHistory, setErrorHistory] = useState<
    { expected: string; pressed: string; tip: string }[]
  >([]);

  // 🧠 Suggest which finger should be used
  const getCorrectionTip = (char: string) => {
    const map: Record<string, string> = {
      A: "Use your left pinky",
      S: "Use your left ring",
      D: "Use your left middle",
      F: "Use your left index",
      G: "Use your left index",
      H: "Use your right index",
      J: "Use your right index",
      K: "Use your right middle",
      L: "Use your right ring",
      ";": "Use your right pinky",
      Q: "Use your left pinky",
      W: "Use your left ring",
      E: "Use your left middle",
      R: "Use your left index",
      T: "Use your left index",
      Y: "Use your right index",
      U: "Use your right index",
      I: "Use your right middle",
      O: "Use your right ring",
      P: "Use your right pinky",
      Z: "Use your left pinky",
      X: "Use your left ring",
      C: "Use your left middle",
      V: "Use your left index",
      B: "Use your left index",
      N: "Use your right index",
      M: "Use your right index",
    };
    return map[char] ? `${map[char]} for "${char}"` : "Check your finger placement";
  };

  const inputRef = useRef<HTMLInputElement>(null);

  const keyboardLayout = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
    ["Z", "X", "C", "V", "B", "N", "M"],
  ];

  // ============================================================
  // Fetch typing data from backend
  // ============================================================
  useEffect(() => {
    const fetchTypingData = async () => {
      try {
        const res = await fetch(`http://localhost:5000/api/typing/level/${level}`);
        const data = await res.json();
        if (data && data.data) {
          const text = data.data.join(" ");
          const wordArray = text.split(" ");
          setWords(wordArray);
          await fetch(`${BASE_URL}/api/detect/set-expected`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ words: wordArray }),
          });
        }
      } catch (error) {
        console.error("Error fetching typing data:", error);
      }
    };
    fetchTypingData();
  }, [level]);

  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, [currentWordIndex]);

  // ============================================================
  // Typing Handlers
  // ============================================================
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    if (value.endsWith(" ")) {
      const typedWord = value.trim();
      setTypedWords((prev) => {
        const updated = [...prev];
        updated[currentWordIndex] = typedWord;
        return updated;
      });
      setCurrentWordIndex((prev) => prev + 1);
      setUserInput("");
      setCharFeedback({});
      return;
    }
    setUserInput(value);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Backspace" || e.key === "Delete") {
      e.preventDefault();
      return;
    }

    const pressedKey = e.key.toUpperCase();
    if (pressedKey === " " || pressedKey === "SPACE") return;
    if (!/^[A-Z]$/.test(pressedKey)) return;

    const currentWord = words[currentWordIndex] || "";
    const typedIndex = userInput.length;
    const expectedChar = currentWord[typedIndex]?.toUpperCase();
    const isCorrect = pressedKey === expectedChar;
    const colorClass = isCorrect ? "bg-green-500 text-white" : "bg-red-500 text-white";

    // ❌ Removed counting here (handled by detection)
    // if (isCorrect) setCorrectCount((prev) => prev + 1);
    // else setIncorrectCount((prev) => prev + 1);

    // ✅ highlight active key
    setActiveKeys((prev) => ({
      ...prev,
      [pressedKey]: colorClass,
    }));

    // ✅ per-character feedback color
    setCharFeedback((prev) => ({
      ...prev,
      [typedIndex]: isCorrect ? "text-green-600" : "text-red-600",
    }));

    // 🧠 UPDATED: Log incorrect key press with ergonomic correction
    if (!isCorrect && expectedChar) {
      const correctionTip = getCorrectionTip(expectedChar);
      setErrorHistory((prev) => {
        const exists = prev.some(
          (err) => err.expected === expectedChar && err.pressed === pressedKey
        );
        return exists
          ? prev
          : [...prev, { expected: expectedChar, pressed: pressedKey, tip: correctionTip }];
      });
    }
  };

  // ============================================================
  // WPM + Accuracy
  // ============================================================
  useEffect(() => {
    const correctCount = typedWords.filter((typed, i) => typed && typed === words[i]).length;
    const totalTyped = typedWords.filter(Boolean).length;
    const accuracyVal = totalTyped > 0 ? Math.round((correctCount / totalTyped) * 100) : 100;
    const timeElapsed = (Date.now() - startTime) / 1000 / 60;
    setAccuracy(accuracyVal);
  }, [typedWords, startTime]);

  // ============================================================
  // Detection Start/Stop
  // ============================================================
  const handleStartDetection = async () => {
    try {
      setIsCalibrating(true);
      setCalibrationDone(false);
      await startDetection();
      setDetecting(true);
      setTimeout(() => {
        setIsCalibrating(false);
        setCalibrationDone(true);
        setTimeout(() => setCalibrationDone(false), 3000);
      }, 10000);
    } catch (err) {
      console.error("Failed to start detection:", err);
      setIsCalibrating(false);
    }
  };

  const handleStopDetection = async () => {
    try {
      await stopDetection();
      setDetecting(false);
    } catch (err) {
      console.error("Failed to stop detection:", err);
    }
  };

  useEffect(() => {
    const start = async () => await handleStartDetection();
    start();
    return () => {
      void handleStopDetection();
    };
  }, []);

  // ============================================================
  // Real-time Detection Sync + Error Logging (deduped)
  // ============================================================
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await getDetectionStatus();
        if (!res || !res.key) return;

        const key = String(res.key).toUpperCase();
        if (key === " " || key === "SPACE") return;

        const expectedKey = res.expected_key ? String(res.expected_key).toUpperCase() : key;
        const isCorrect = res.correct === true;

        // Build unique signature for deduplication
        const signature = JSON.stringify({
          key,
          expectedKey,
          correct: isCorrect,
          ts: res.ts || res.time || res.event_id || null,
        });

        if (lastEventRef.current === signature) return;
        lastEventRef.current = signature;

        // ✅ Highlight detected key
        setActiveKeys((prev) => ({
          ...prev,
          [key]: isCorrect ? "bg-green-500 text-white" : "bg-red-500 text-white",
        }));

        // ✅ Update counters once per new detection
        if (isCorrect) setCorrectCount((prev) => prev + 1);
        else setIncorrectCount((prev) => prev + 1);

        setLastKey(key);

        // 🧠 NEW: Capture incorrect detections
        if (!isCorrect && expectedKey && key !== expectedKey) {
          const correctionTip = getCorrectionTip(expectedKey);
          setErrorHistory((prev) => {
            const exists = prev.some(
              (err) => err.expected === expectedKey && err.pressed === key
            );
            if (exists) return prev;
            const updated = [
              ...prev,
              { expected: expectedKey, pressed: key, tip: correctionTip },
            ];
            return updated.slice(-50); // keep last 50 errors max
          });
        }
      } catch (err) {
        console.error("Detection sync error:", err);
      }
    }, 300);

    return () => clearInterval(interval);
  }, []); // ✅ Run once only

  // ============================================================
  // Render top letters (purple highlight)
  // ============================================================
  const renderWord = (word: string, index: number) => {
    const typed = index === currentWordIndex ? userInput : typedWords[index] || "";
    const isCurrent = index === currentWordIndex;

    return (
      <span key={index} className="mr-3">
        {word.split("").map((char, i) => {
          let color = "text-muted-foreground";
          if (isCurrent && i === typed.length) {
            color = "text-purple-500 underline animate-pulse";
          }
          return (
            <span key={i} className={`font-pixel ${color}`}>
              {char}
            </span>
          );
        })}
      </span>
    );
  };

  // ============================================================
  // Generate Standard-Based Feedback (ISO 9241-410 / 960)
  // ============================================================
  const generateStandardFeedback = (wpm: number, accuracy: number, correct: number, incorrect: number) => {
    let feedbackLines: string[] = [];

    // --- ISO 9241-410 (Ergonomic keyboard interaction)
    if (accuracy >= 95) {
      feedbackLines.push("🎯 [Accuracy] 🌟 Awesome job! Your typing is super accurate. Keep it up!");
    } else if (accuracy >= 85) {
      feedbackLines.push("🎯 [Accuracy] 👍 Great effort! Try to keep your fingers steady and aim for fewer mistakes.");
    } else {
      feedbackLines.push("🎯 [Accuracy] 💪 You're doing okay! Slow down a bit and watch which keys your fingers press.");
    }

    // --- ISO 9241-960 (Performance & usability)
    if (wpm >= 40) {
      feedbackLines.push("⚡ [Speed] 🚀 You type really fast! Keep practicing to stay smooth and steady.");
    } else if (wpm >= 25) {
      feedbackLines.push("⚡ [Speed] 🏃 Nice speed! You’re getting quicker—just keep your hands relaxed.");
    } else {
      feedbackLines.push("⚡ [Speed] 🐢 Go slowly and carefully. Accuracy first, speed will come later!");
    }

    // --- Finger precision insight
    const total = correct + incorrect;
    const errorRate = total > 0 ? (incorrect / total) * 100 : 0;
    if (errorRate > 20) {
      feedbackLines.push("🎮 [Finger Control] ✋ Oops! Some fingers are mixing up. Try to remember which finger goes with each key!");
    } else if (errorRate > 10) {
      feedbackLines.push("🎮 [Finger Control] 🖐 You're close! Focus on using the right fingers for each letter.");
    } else {
      feedbackLines.push("🎮 [Finger Control]👏 Great finger control! You’re typing like a pro!");
    }

    return feedbackLines;
  };

  // ============================================================
  // Finish Session (Phase 1 Metrics Foundation)
  // ============================================================
  const handleFinish = async () => {
    const sessionDuration = Math.round((Date.now() - startTime) / 1000);

    const report: SessionReport = {
      wpm,
      accuracy,
      correct_keystrokes: correctCount,
      incorrect_keystrokes: incorrectCount,
      fingerAccuracy,
      timingVariance,
      session_duration_sec: sessionDuration,
    };

    console.log("📊 Session Report:", report);

    if (sessionType === "evaluated") {
      const grade =
        wpm >= 40 && accuracy >= 95
          ? "A"
          : wpm >= 30 && accuracy >= 85
          ? "B+"
          : wpm >= 20 && accuracy >= 75
          ? "B"
          : "C";

      try {
        await fetch("http://localhost:5000/api/results", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ level, wpm, accuracy, grade, sessionType }),
        });
        console.log("✅ Session result saved to MongoDB");
      } catch (error) {
        console.error("❌ Failed to save result:", error);
      }
    }
    navigate("/student/play");
  };

  const isFinished = currentWordIndex >= words.length;

  // ============================================================
  // UI
  // ============================================================
  return (
    <div className="min-h-screen p-8 flex flex-col items-center">
      {(isCalibrating || calibrationDone) && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-50">
          <PixelCard className="p-8 text-center">
            {isCalibrating ? (
              <>
                <p className="font-pixel text-lg text-yellow-400 mb-2">
                  🔧 Calibrating Keyboard Layout...
                </p>
                <p className="font-pixel text-sm text-muted-foreground">
                  Please remove your hands from the keyboard.
                </p>
              </>
            ) : (
              <p className="font-pixel text-lg text-green-500">✅ Calibration Complete!</p>
            )}
          </PixelCard>
        </div>
      )}

      <div className="max-w-7xl w-full relative">
        {/* Header */}
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
            <p className="font-pixel text-xs mb-1">Correct (Finger)</p>
            <p className="font-pixel text-2xl">{correctCount}</p>
          </PixelCard>
          <PixelCard variant="red">
            <p className="font-pixel text-xs mb-1">Incorrect (Finger)</p>
            <p className="font-pixel text-2xl">{incorrectCount}</p>
          </PixelCard>
        </div>

        {/* Typing Area */}
        {!isFinished ? (
          <PixelCard className="mb-8 flex flex-col items-center justify-center text-center py-8">
            <div className="font-pixel text-lg mb-6 flex flex-wrap justify-center gap-2 max-w-3xl leading-relaxed">
              {words.map((word, index) => renderWord(word, index))}
            </div>

            <input
              ref={inputRef}
              type="text"
              value={userInput}
              onChange={handleChange}
              onKeyDown={handleKeyDown}
              className="text-center w-2/3 md:w-1/2 px-4 py-3 bg-input border-[3px] border-border text-foreground font-pixel text-lg focus:outline-none focus:ring-2 focus:ring-primary rounded-md"
              placeholder="Type here..."
              autoFocus
            />

            <div className="font-pixel text-sm text-muted-foreground mt-6 mb-2">
              Keyboard (60% Layout)
            </div>
            <div className="flex flex-col items-center gap-2">
              {keyboardLayout.map((row, rowIdx) => (
                <div key={rowIdx} className="flex gap-2 justify-center">
                  {row.map((key) => (
                    <div
                      key={key}
                      className={`pixel-border w-10 h-10 flex items-center justify-center font-pixel text-sm border border-border rounded-md ${
                        activeKeys[key] ? activeKeys[key] : "bg-muted text-foreground"
                      }`}
                    >
                      {key}
                    </div>
                  ))}
                </div>
              ))}
              <div className="pixel-border w-[300px] h-10 flex items-center justify-center font-pixel text-sm border border-border rounded-md mt-2 bg-muted text-foreground">
                SPACE
              </div>
            </div>
          </PixelCard>
        ) : (
          <PixelCard className="text-center py-8">
            <p className="font-pixel text-xl text-green-600 mb-4">🎉 Session Complete!</p>
            <div className="flex flex-col items-center gap-2 font-pixel text-sm text-muted-foreground">
              <p>
                You typed <span className="text-primary">{typedWords.length}</span> words
                with <span className="text-green-500">{accuracy}%</span> accuracy at
                <span className="text-yellow-500"> {wpm} WPM</span>.
              </p>

              <hr className="border-border w-1/2 my-2" />

              {generateStandardFeedback(wpm, accuracy, correctCount, incorrectCount).map((line, idx) => (
                <p key={idx} className="text-center leading-relaxed">
                  {line}
                </p>
              ))}

              {/* ============================================================ */}
              {/* 🧠 Common Mistakes Section (Error History Recap) */}
              {/* ============================================================ */}
              {errorHistory.length > 0 && (
                <>
                  <hr className="border-border w-1/2 my-3" />
                  <p className="font-pixel text-md text-red-500 mb-2">❌ Common Mistakes</p>
                  <div className="flex flex-col gap-1 items-center text-xs text-muted-foreground">
                    {errorHistory.slice(-5).map((err, idx) => (
                      <p key={idx}>
                        You pressed <span className="text-red-500">{err.pressed}</span> instead of{" "}
                        <span className="text-green-500">{err.expected}</span> → {err.tip}
                      </p>
                    ))}
                  </div>
                </>
              )}
            </div>
          </PixelCard>
        )}

        {/* Detection + Finish Buttons */}
        <div className="flex justify-center mt-6 gap-4">
          {!detecting ? (
            <PixelButton variant="orange" size="lg" onClick={handleStartDetection}>
              🎥 Start Detection
            </PixelButton>
          ) : (
            <PixelButton variant="red" size="lg" onClick={handleStopDetection}>
              🛑 Stop Detection
            </PixelButton>
          )}
          <PixelButton variant="primary" size="lg" onClick={handleFinish}>
            {isFinished ? "Back to Play" : "Finish Session"}
          </PixelButton>
        </div>
      </div>
    </div>
  );
}
