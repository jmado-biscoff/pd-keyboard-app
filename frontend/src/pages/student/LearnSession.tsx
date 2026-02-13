import { useState, useEffect, useRef } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { Logo } from "@/components/Logo";
import { PixelButton } from "@/components/PixelButton";
import { PixelCard } from "@/components/PixelCard";
import { ArrowLeft } from "lucide-react";
import { CalibrationOverlay } from "@/components/CalibrationOverlay";
import { VirtualKeyboard } from "@/components/VirtualKeyboard";
import { VideoFeed } from "@/components/VideoFeed";
import { KEYBOARD_IMAGES } from "@/utils/keyboardImages";
import {
  startDetection,
  stopDetection,
  getKeyColor,
  getCorrectionTip,
} from "@/utils/typingHelpers";
import bgVideo from "@/assets/b4.mp4";

const BASE_URL = import.meta.env.VITE_API_URL.replace("/api/auth", "");

const MODULE_TITLES: Record<number, string> = {
  1: "Home Row Heroes",
  2: "Top Row Adventure",
  3: "Bottom Row Explorer",
  4: "Alphabet Mastery",
  5: "Word Builder",
};

const MODULE_DESCRIPTIONS: Record<number, string> = {
  1: "Master the home row keys: A S D F J K L",
  2: "Learn the top row: Q W E R T Y U I O P",
  3: "Practice the bottom row: Z X C V B N M",
  4: "Full alphabet integration: all A-Z",
  5: "Build words with rhythm and flow",
};

export default function LearnSession() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const moduleId = parseInt(searchParams.get("module") || "1");

  // Content
  const [drills, setDrills] = useState<string[]>([]);
  const [currentDrillIndex, setCurrentDrillIndex] = useState(0);
  const [currentCharIndex, setCurrentCharIndex] = useState(0);
  const [charFeedback, setCharFeedback] = useState<Record<number, "correct" | "incorrect">>({});

  // Detection state
  const [detecting, setDetecting] = useState(false);
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [calibrationDone, setCalibrationDone] = useState(false);
  const [showCalibrationComplete, setShowCalibrationComplete] = useState(false);
  const [calibrationProgress, setCalibrationProgress] = useState({ detected: 0, required: 26 });
  const [calibratedKeys, setCalibratedKeys] = useState<string[]>([]);
  const [detectionError, setDetectionError] = useState<string | null>(null);
  const [frame, setFrame] = useState<string | null>(null);

  // Feedback
  const [activeKeys, setActiveKeys] = useState<Record<string, string>>({});
  const [correctCount, setCorrectCount] = useState(0);
  const [incorrectCount, setIncorrectCount] = useState(0);
  const [lastTip, setLastTip] = useState<string | null>(null);

  // Completion
  const [moduleComplete, setModuleComplete] = useState(false);

  // Refs for SSE handler (avoid stale closures)
  const drillsRef = useRef<string[]>([]);
  const currentDrillIndexRef = useRef(0);
  const currentCharIndexRef = useRef(0);
  const correctCountRef = useRef(0);
  const incorrectCountRef = useRef(0);
  const isCalibratingRef = useRef(false);
  const calibrationDoneRef = useRef(false);
  const lastEventRef = useRef("");
  const calibrationTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Sync refs
  useEffect(() => { drillsRef.current = drills; }, [drills]);
  useEffect(() => { currentDrillIndexRef.current = currentDrillIndex; }, [currentDrillIndex]);
  useEffect(() => { currentCharIndexRef.current = currentCharIndex; }, [currentCharIndex]);
  useEffect(() => { isCalibratingRef.current = isCalibrating; }, [isCalibrating]);

  // Focus management
  useEffect(() => {
    const focusInput = () => {
      if (inputRef.current && !moduleComplete) inputRef.current.focus();
    };
    focusInput();
    document.addEventListener("click", focusInput);
    return () => document.removeEventListener("click", focusInput);
  }, [moduleComplete]);

  // Fetch drills from backend
  useEffect(() => {
    const fetchDrills = async () => {
      try {
        const res = await fetch(`${BASE_URL}/api/learn/module/${moduleId}`);
        const data = await res.json();
        if (data && data.drills) {
          setDrills(data.drills);
          // Send expected keys to detection backend
          await fetch(`${BASE_URL}/api/detect/set-expected`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              words: data.drills,
              startIndex: 0,
            }),
          });
        }
      } catch (error) {
        console.error("Error fetching learn drills:", error);
      }
    };
    fetchDrills();
  }, [moduleId]);

  // Auto-dismiss calibration complete overlay
  useEffect(() => {
    if (showCalibrationComplete) {
      const t = setTimeout(() => setShowCalibrationComplete(false), 2000);
      return () => clearTimeout(t);
    }
  }, [showCalibrationComplete]);

  // ============================================================
  // SSE Connection for detection events
  // ============================================================
  useEffect(() => {
    if (!detecting || moduleComplete) return;

    const source = new EventSource(`${BASE_URL}/api/detect/stream`);

    source.onmessage = (event: MessageEvent<string>) => {
      try {
        const data = JSON.parse(event.data);

        switch (data.type) {
          case "calibration_progress":
            setFrame(data.frame || null);
            setCalibrationProgress({
              detected: data.detected,
              required: data.required,
            });
            if (Array.isArray(data.detected_keys)) {
              setCalibratedKeys(data.detected_keys.map((k: string) => k.toUpperCase()));
            }
            break;

          case "calibration_done": {
            if (!calibrationDoneRef.current) {
              if (calibrationTimeoutRef.current) {
                clearTimeout(calibrationTimeoutRef.current);
                calibrationTimeoutRef.current = null;
              }

              const keysDetected = data.locked_keys || 0;
              if (keysDetected < 26) {
                setDetectionError("Calibration incomplete - not all keys detected. Please realign your keyboard.");
                setIsCalibrating(false);
                return;
              }

              calibrationDoneRef.current = true;
              setCalibrationDone(true);
              setIsCalibrating(false);
              setShowCalibrationComplete(true);
            }
            break;
          }

          case "error":
            setDetectionError(data.message);
            setDetecting(false);
            if (calibrationTimeoutRef.current) {
              clearTimeout(calibrationTimeoutRef.current);
              calibrationTimeoutRef.current = null;
            }
            break;

          case "frame":
            setFrame(data.frame || null);
            break;

          case "detection": {
            if (!data.key) return;
            const key = String(data.key).toUpperCase();
            if (!/^[A-Z]$/.test(key)) return;
            if (isCalibratingRef.current) return;

            const currentDrills = drillsRef.current;
            const dIdx = currentDrillIndexRef.current;
            const cIdx = currentCharIndexRef.current;

            if (dIdx >= currentDrills.length) return;
            const currentDrill = currentDrills[dIdx];
            if (!currentDrill || cIdx >= currentDrill.length) return;

            const expectedChar = currentDrill[cIdx].toUpperCase();

            // Duplicate prevention
            const signature = `${key}-${dIdx}-${cIdx}`;
            if (lastEventRef.current === signature) return;
            lastEventRef.current = signature;

            const expectedKey = data.expected_key ? String(data.expected_key).toUpperCase() : expectedChar;
            const mlCorrect = data.ml_label === "Correct";
            const isCorrect = mlCorrect && key === expectedChar;

            // Update counts
            if (isCorrect) {
              correctCountRef.current++;
              setCorrectCount(correctCountRef.current);
              setLastTip(null);
            } else {
              incorrectCountRef.current++;
              setIncorrectCount(incorrectCountRef.current);
              setLastTip(getCorrectionTip(expectedChar));
            }

            // Visual feedback on VirtualKeyboard
            const color = getKeyColor(key, expectedChar, data.ml_label);
            setActiveKeys({ [key]: color });
            setTimeout(() => setActiveKeys({}), 300);

            // Update char feedback for drill display
            setCharFeedback((prev) => ({
              ...prev,
              [cIdx]: isCorrect ? "correct" : "incorrect",
            }));

            // Advance character pointer
            const nextCharIndex = cIdx + 1;
            if (nextCharIndex >= currentDrill.length) {
              // Drill complete — advance to next
              const nextDrillIndex = dIdx + 1;
              if (nextDrillIndex >= currentDrills.length) {
                setModuleComplete(true);
              } else {
                currentDrillIndexRef.current = nextDrillIndex;
                currentCharIndexRef.current = 0;
                setCurrentDrillIndex(nextDrillIndex);
                setCurrentCharIndex(0);
                setCharFeedback({});
                setLastTip(null);

                // Update expected keys for new drill
                fetch(`${BASE_URL}/api/detect/set-expected`, {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({
                    words: currentDrills.slice(nextDrillIndex),
                    startIndex: 0,
                  }),
                });
              }
            } else {
              currentCharIndexRef.current = nextCharIndex;
              setCurrentCharIndex(nextCharIndex);
            }
            break;
          }
        }
      } catch (err) {
        console.error("LearnSession SSE parse error:", err);
      }
    };

    source.onerror = () => {
      if (source.readyState === EventSource.CLOSED) {
        console.error("LearnSession SSE connection closed");
      }
    };

    return () => {
      source.close();
    };
  }, [detecting, moduleComplete]);

  // ============================================================
  // Start detection on mount
  // ============================================================
  const handleStartDetection = async () => {
    try {
      setFrame(null);
      setCalibrationProgress({ detected: 0, required: 26 });
      setActiveKeys({});
      setCalibratedKeys([]);
      setDetectionError(null);
      setIsCalibrating(true);
      setCalibrationDone(false);
      setShowCalibrationComplete(false);
      calibrationDoneRef.current = false;

      if (calibrationTimeoutRef.current) {
        clearTimeout(calibrationTimeoutRef.current);
      }

      // Send expected keys before starting detection
      if (drillsRef.current.length > 0) {
        await fetch(`${BASE_URL}/api/detect/set-expected`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            words: drillsRef.current,
            startIndex: 0,
          }),
        });
      }

      calibrationTimeoutRef.current = setTimeout(() => {
        if (!calibrationDoneRef.current) {
          setDetectionError("No keyboard detected. Please ensure your keyboard is visible to the camera.");
          setIsCalibrating(false);
        }
      }, 15000);

      await startDetection();
      setDetecting(true);
    } catch (err) {
      console.error("Failed to start detection:", err);
      setIsCalibrating(false);
    }
  };

  const handleStopDetection = async () => {
    try {
      await stopDetection();
    } catch (err) {
      console.error("Failed to stop detection:", err);
    } finally {
      setDetecting(false);
    }
  };

  // Recalibrate
  const handleRecalibrate = async () => {
    try {
      setFrame(null);
      setCalibrationProgress({ detected: 0, required: 26 });
      setActiveKeys({});
      setCalibratedKeys([]);
      setCalibrationDone(false);
      setShowCalibrationComplete(false);
      setIsCalibrating(true);
      calibrationDoneRef.current = false;
      setDetecting(false);

      if (detecting) await stopDetection();

      // Cool-down for clean camera release
      await new Promise((r) => setTimeout(r, 1000));
      await handleStartDetection();
    } catch (err) {
      console.error("Recalibration failed:", err);
      setIsCalibrating(false);
    }
  };

  // Auto-start detection on mount
  useEffect(() => {
    const start = async () => await handleStartDetection();
    start();
    return () => {
      void handleStopDetection();
      if (calibrationTimeoutRef.current) clearTimeout(calibrationTimeoutRef.current);
    };
  }, []);

  // Tab = Recalibrate shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (moduleComplete) return;
      if (e.key === "Tab") {
        e.preventDefault();
        handleRecalibrate();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [moduleComplete, detecting]);

  // Finish module
  const handleFinish = () => {
    const saved = JSON.parse(localStorage.getItem("typingModuleProgress") || "{}");
    saved[moduleId] = { completed: true };
    localStorage.setItem("typingModuleProgress", JSON.stringify(saved));
    navigate("/student/learn");
  };

  // Current drill data
  const currentDrill = drills[currentDrillIndex] || "";
  const targetChar = currentDrill[currentCharIndex]?.toUpperCase();
  const keyboardImage = targetChar ? KEYBOARD_IMAGES[targetChar] : null;
  const totalChars = correctCount + incorrectCount;
  const accuracyPercent = totalChars > 0 ? Math.round((correctCount / totalChars) * 100) : 100;

  return (
    <div className="relative min-h-screen overflow-hidden">
      {/* Background */}
      <video autoPlay loop muted playsInline className="absolute top-0 left-0 w-full h-full object-cover -z-10">
        <source src={bgVideo} type="video/mp4" />
      </video>

      {/* Calibration Overlay */}
      <CalibrationOverlay
        isCalibrating={isCalibrating}
        showCalibrationComplete={showCalibrationComplete}
        calibrationProgress={calibrationProgress}
        calibratedKeys={calibratedKeys}
      />

      {/* Detection Error Overlay */}
      {detectionError && (
        <div className="fixed inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm z-50">
          <PixelCard className="p-8 text-center max-w-md w-full mx-4">
            <p className="font-pixel text-lg text-red-400 mb-4">Detection Error</p>
            <p className="font-pixel text-xs text-muted-foreground mb-6">{detectionError}</p>
            <div className="flex gap-3 justify-center">
              <PixelButton variant="secondary" onClick={() => navigate("/student/learn")}>
                Back to Modules
              </PixelButton>
              <PixelButton variant="primary" onClick={() => { setDetectionError(null); handleRecalibrate(); }}>
                Retry
              </PixelButton>
            </div>
          </PixelCard>
        </div>
      )}

      {/* Page Content */}
      <div className="relative z-10 p-4 bg-black/20 min-h-screen">
        <div className="max-w-[1200px] mx-auto">
          {/* Header */}
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <PixelButton variant="secondary" onClick={() => { handleStopDetection(); navigate("/student/learn"); }}>
                <ArrowLeft size={18} />
              </PixelButton>
              <Logo />
            </div>
            <h1 className="font-pixel text-lg text-black">
              Module {moduleId}: {MODULE_TITLES[moduleId]}
            </h1>
          </div>

          {!moduleComplete ? (
            <>
              {/* Main Layout: Video + Learning Interface */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
                {/* Left: Video Feed */}
                <div>
                  <VideoFeed
                    detecting={detecting}
                    calibrationDone={calibrationDone}
                    baseUrl={BASE_URL}
                  />
                </div>

                {/* Right: Learning Interface */}
                <div className="flex flex-col items-center justify-center gap-4">
                  {/* Module description */}
                  <p className="font-pixel text-xs text-black/70 text-center">
                    {MODULE_DESCRIPTIONS[moduleId]}
                  </p>

                  {/* Proper keyboard image for target letter */}
                  {keyboardImage && calibrationDone && (
                    <div className="flex justify-center">
                      <img
                        src={keyboardImage}
                        alt={`Press ${targetChar}`}
                        className="w-full max-w-md h-auto rounded-xl shadow-2xl border-2 border-yellow-300/60"
                      />
                    </div>
                  )}

                  {/* Current target letter */}
                  {calibrationDone && targetChar && (
                    <div className="text-center">
                      <p className="font-pixel text-[10px] text-black/50 uppercase tracking-widest mb-1">
                        Press this key
                      </p>
                      <p className="font-pixel text-5xl text-purple-600 animate-pulse drop-shadow-lg">
                        {targetChar}
                      </p>
                    </div>
                  )}

                  {/* Drill display with character-level feedback */}
                  {calibrationDone && currentDrill && (
                    <PixelCard className="w-full max-w-md bg-black/60 border-2 border-yellow-300 backdrop-blur-sm">
                      <div className="font-pixel text-xl text-center tracking-[0.3em] py-2">
                        {currentDrill.split("").map((ch, i) => {
                          let colorClass = "text-gray-500";
                          if (charFeedback[i] === "correct") colorClass = "text-green-400";
                          else if (charFeedback[i] === "incorrect") colorClass = "text-red-400";
                          else if (i === currentCharIndex) colorClass = "text-purple-400 animate-pulse";
                          return (
                            <span key={i} className={colorClass}>
                              {ch.toUpperCase()}
                            </span>
                          );
                        })}
                      </div>
                    </PixelCard>
                  )}

                  {/* Correction tip */}
                  {lastTip && calibrationDone && (
                    <p className="font-pixel text-xs text-red-400 text-center">
                      {lastTip}
                    </p>
                  )}

                  {/* Progress & Metrics */}
                  {calibrationDone && (
                    <div className="flex gap-4">
                      <PixelCard variant="yellow" className="text-center px-4 py-2">
                        <p className="font-pixel text-[9px] text-black/60">Drill</p>
                        <p className="font-pixel text-lg text-black">{currentDrillIndex + 1}/{drills.length}</p>
                      </PixelCard>
                      <PixelCard variant="green" className="text-center px-4 py-2">
                        <p className="font-pixel text-[9px] text-black/60">Accuracy</p>
                        <p className="font-pixel text-lg text-black">{accuracyPercent}%</p>
                      </PixelCard>
                      <PixelCard variant="purple" className="text-center px-4 py-2">
                        <p className="font-pixel text-[9px]">Correct</p>
                        <p className="font-pixel text-lg">{correctCount}</p>
                      </PixelCard>
                    </div>
                  )}
                </div>
              </div>

              {/* Virtual Keyboard */}
              <div className="mb-3">
                <VirtualKeyboard activeKeys={activeKeys} isCalibrating={isCalibrating} />
              </div>

              {/* Recalibrate button */}
              <div className="text-center">
                <PixelButton
                  variant="secondary"
                  size="sm"
                  onClick={handleRecalibrate}
                  disabled={isCalibrating}
                >
                  Recalibrate (Tab)
                </PixelButton>
              </div>

              {/* Hidden input for focus */}
              <input
                ref={inputRef}
                type="text"
                className="absolute opacity-0 w-0 h-0 focus:outline-none"
                autoFocus
                aria-label="Learn typing input"
                onKeyDown={(e) => {
                  if (e.key === "Backspace" || e.key === "Delete") e.preventDefault();
                }}
              />
            </>
          ) : (
            /* Module Complete Screen */
            <div className="flex items-center justify-center min-h-[60vh]">
              <PixelCard className="p-8 text-center max-w-lg w-full bg-black/70 border-2 border-yellow-300 backdrop-blur-md">
                <h2 className="font-pixel text-2xl text-yellow-200 mb-4">Module Complete!</h2>
                <p className="font-pixel text-sm text-white mb-2">
                  {MODULE_TITLES[moduleId]}
                </p>
                <p className="font-pixel text-xs text-white/70 mb-6">
                  {accuracyPercent >= 90
                    ? "Excellent work! You're mastering these keys!"
                    : accuracyPercent >= 70
                    ? "Good job! Keep practicing for even better accuracy."
                    : "Keep practicing! Focus on using the correct fingers."}
                </p>
                <div className="grid grid-cols-3 gap-4 mb-6">
                  <PixelCard className="bg-black/50 border border-yellow-200 text-center">
                    <p className="font-pixel text-[9px] text-yellow-200 mb-1">Accuracy</p>
                    <p className="font-pixel text-xl text-white">{accuracyPercent}%</p>
                  </PixelCard>
                  <PixelCard className="bg-black/50 border border-green-400/50 text-center">
                    <p className="font-pixel text-[9px] text-green-300 mb-1">Correct</p>
                    <p className="font-pixel text-xl text-white">{correctCount}</p>
                  </PixelCard>
                  <PixelCard className="bg-black/50 border border-red-400/50 text-center">
                    <p className="font-pixel text-[9px] text-red-300 mb-1">Incorrect</p>
                    <p className="font-pixel text-xl text-white">{incorrectCount}</p>
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
