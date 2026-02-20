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
import { getKeyColor, getCorrectionTip } from "@/utils/typingHelpers";
import { runCalibration } from "@/utils/calibrationClient";
import {
  initModels,
  startClientDetection,
  stopClientDetection,
  disposeAll,
} from "@/utils/detectionOrchestrator";
import { useAudio } from "@/contexts/AudioContext";
import bgVideo from "@/assets/bg2.mp4";

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

  // Audio
  const { playCorrectSound, playErrorSound, muteMusicTemporarily } = useAudio();

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

  // Refs for detection handler (avoid stale closures)
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

  // Client-side detection refs
  const calibrationStopRef = useRef<(() => void) | null>(null);
  const keyPositionsRef = useRef<Record<string, number[]>>({});
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const modelsInitializedRef = useRef(false);

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
  // Detection event handler (called by orchestrator callbacks)
  // ============================================================
  const handleDetectionEvent = (data: {
    type: "detection";
    key: string;
    finger: string;
    hand: string;
    expected_key: string | null;
    ml_label: "Correct" | "Incorrect";
    fingertip_count: number;
    left_fingers_count: number;
    right_fingers_count: number;
  }) => {
    if (!data.key) return;
    const key = String(data.key).toUpperCase();
    if (!/^[A-Z]$/.test(key)) return;
    if (isCalibratingRef.current) return;

    const currentDrills = drillsRef.current;
    const dIdx = currentDrillIndexRef.current;
    const cIdx = currentCharIndexRef.current;

    if (dIdx >= currentDrills.length) return;
    const currentDrillText = currentDrills[dIdx];
    if (!currentDrillText || cIdx >= currentDrillText.length) return;

    const expectedChar = currentDrillText[cIdx].toUpperCase();

    // Duplicate prevention
    const signature = `${key}-${dIdx}-${cIdx}`;
    if (lastEventRef.current === signature) return;
    lastEventRef.current = signature;

    const mlCorrect = data.ml_label === "Correct";
    const keyCorrect = key === expectedChar;
    const isFullyCorrect = mlCorrect && keyCorrect;

    // Visual feedback on VirtualKeyboard
    const color = getKeyColor(key, expectedChar, data.ml_label);
    setActiveKeys({ [key]: color });
    setTimeout(() => setActiveKeys({}), 300);

    if (isFullyCorrect) {
      correctCountRef.current++;
      setCorrectCount(correctCountRef.current);
      setLastTip(null);
      playCorrectSound();

      setCharFeedback((prev) => ({ ...prev, [cIdx]: "correct" }));

      const nextCharIndex = cIdx + 1;
      if (nextCharIndex >= currentDrillText.length) {
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
        }
      } else {
        currentCharIndexRef.current = nextCharIndex;
        setCurrentCharIndex(nextCharIndex);
      }
    } else {
      incorrectCountRef.current++;
      setIncorrectCount(incorrectCountRef.current);
      playErrorSound();

      if (!keyCorrect && !mlCorrect) {
        setLastTip(`Wrong key AND wrong finger! Expected: ${expectedChar}. ${getCorrectionTip(expectedChar)}`);
      } else if (!keyCorrect) {
        setLastTip(`Wrong key! Expected: ${expectedChar}`);
      } else if (!mlCorrect) {
        setLastTip(`Wrong finger! ${getCorrectionTip(expectedChar)}`);
      }

      setCharFeedback((prev) => ({ ...prev, [cIdx]: "incorrect" }));

      setTimeout(() => {
        setCharFeedback((prev) => {
          const updated = { ...prev };
          delete updated[cIdx];
          return updated;
        });
      }, 500);
    }
  };

  // Start client-side detection after calibration completes and VideoFeed renders detection refs
  useEffect(() => {
    if (!calibrationDone || !detecting) return;
    if (!videoRef.current || !canvasRef.current) return;
    if (Object.keys(keyPositionsRef.current).length === 0) return;

    const startDetect = async () => {
      try {
        await startClientDetection(
          keyPositionsRef.current,
          {
            onDetection: handleDetectionEvent,
            onFrame: () => {},
            onError: (err) => { setDetectionError(err.message); setDetecting(false); },
          },
          videoRef.current!,
          canvasRef.current!
        );
      } catch (err) {
        console.error("Failed to start client detection:", err);
        setDetectionError("Failed to start typing detection. Please retry.");
      }
    };
    startDetect();
  }, [calibrationDone, detecting]);

  // ============================================================
  // Start detection (calibration → client-side detection)
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

      // Initialize ML models (SVM + MediaPipe) in parallel
      if (!modelsInitializedRef.current) {
        await initModels();
        modelsInitializedRef.current = true;
      }

      // Create temporary video element for calibration
      const tempVideo = document.createElement("video");
      tempVideo.autoplay = true;
      tempVideo.playsInline = true;
      tempVideo.muted = true;
      tempVideo.style.display = "none";
      document.body.appendChild(tempVideo);

      const { stop, stream } = await runCalibration(
        tempVideo,
        // onProgress
        (progress) => {
          setFrame(progress.annotated_frame || null);
          setCalibrationProgress({
            detected: progress.detected,
            required: progress.required,
          });
          if (Array.isArray(progress.detected_keys)) {
            setCalibratedKeys(progress.detected_keys.map((k: string) => k.toUpperCase()));
          }
        },
        // onComplete
        async (keyPositions) => {
          if (calibrationTimeoutRef.current) {
            clearTimeout(calibrationTimeoutRef.current);
            calibrationTimeoutRef.current = null;
          }

          // Stop calibration camera
          stream.getTracks().forEach((t) => t.stop());
          tempVideo.remove();
          calibrationStopRef.current = null;

          keyPositionsRef.current = keyPositions;
          calibrationDoneRef.current = true;
          setCalibrationDone(true);
          setIsCalibrating(false);
          setShowCalibrationComplete(true);
          setDetecting(true);

          // Client detection will be started by useEffect after VideoFeed re-renders with detection refs
        },
        // onError
        (msg) => {
          stream.getTracks().forEach((t) => t.stop());
          tempVideo.remove();
          calibrationStopRef.current = null;
          setDetectionError(msg);
          setIsCalibrating(false);
        }
      );

      calibrationStopRef.current = stop;
    } catch (err) {
      console.error("Failed to start detection:", err);
      setIsCalibrating(false);
      setDetectionError("Failed to start camera. Please check permissions.");
    }
  };

  const handleStopDetection = () => {
    if (calibrationStopRef.current) {
      calibrationStopRef.current();
      calibrationStopRef.current = null;
    }
    stopClientDetection();
    setDetecting(false);
  };

  // Recalibrate
  const handleRecalibrate = async () => {
    setFrame(null);
    setCalibrationProgress({ detected: 0, required: 26 });
    setActiveKeys({});
    setCalibratedKeys([]);
    setCalibrationDone(false);
    setShowCalibrationComplete(false);
    calibrationDoneRef.current = false;

    if (calibrationStopRef.current) {
      calibrationStopRef.current();
      calibrationStopRef.current = null;
    }
    stopClientDetection();
    setDetecting(false);

    // Brief cool-down for clean camera release
    await new Promise((r) => setTimeout(r, 500));
    await handleStartDetection();
  };

  // Auto-start detection on mount
  useEffect(() => {
    muteMusicTemporarily(true);

    const start = async () => await handleStartDetection();
    start();

    return () => {
      try {
        muteMusicTemporarily(false);
        disposeAll(); // Full cleanup: stop detection + dispose ML models on unmount
        if (calibrationStopRef.current) calibrationStopRef.current();
        if (calibrationTimeoutRef.current) clearTimeout(calibrationTimeoutRef.current);
      } catch (err) {
        console.error("LearnSession cleanup error:", err);
      }
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
  const handleFinish = async () => {
    const token = localStorage.getItem("token");

    if (token) {
      try {
        await fetch(`${BASE_URL}/api/student/learning-progress`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({
            moduleId,
            completed: true,
            accuracy: accuracyPercent,
          }),
        });
      } catch (error) {
        console.error("Failed to save progress to backend:", error);
      }
    }

    const saved = JSON.parse(localStorage.getItem("typingModuleProgress") || "{}");
    saved[moduleId] = { completed: true, accuracy: accuracyPercent };
    localStorage.setItem("typingModuleProgress", JSON.stringify(saved));

    navigate("/student/learn");
  };

  // Current drill data
  const currentDrill = drills[currentDrillIndex] || "";
  const targetChar = currentDrill[currentCharIndex]?.toUpperCase();
  const keyboardImage = targetChar ? KEYBOARD_IMAGES[targetChar] : null;
  const totalChars = correctCount + incorrectCount;
  const accuracyPercent = totalChars > 0 ? Math.round((correctCount / totalChars) * 100) : 100;

  // Determine VideoFeed mode
  const videoFeedMode = isCalibrating ? "calibration" : calibrationDone && detecting ? "detection" : "idle";

  return (
    <div className="min-h-screen flex items-start justify-center p-4 pt-6 relative">
      {/* Background */}
      <video autoPlay loop muted playsInline className="fixed top-0 left-0 w-full h-full object-cover -z-10">
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

      {/* Main Content Card */}
      <div className="w-full max-w-[1200px] rounded-xl border border-white/20 shadow-2xl bg-black/30 backdrop-blur-md overflow-hidden flex flex-col">
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-white/20 bg-black/40 backdrop-blur-sm shrink-0">
            <div className="flex items-center gap-3">
              <PixelButton variant="secondary" size="sm" onClick={() => { handleStopDetection(); navigate("/student/learn"); }}>
                <ArrowLeft size={18} />
              </PixelButton>
              <Logo />
            </div>
            <div className="bg-black/60 border-2 border-yellow-400 rounded-lg px-4 py-2 backdrop-blur-sm">
              <h1 className="font-pixel text-lg text-yellow-300">
                Module {moduleId}: {MODULE_TITLES[moduleId]}
              </h1>
            </div>
          </div>

          {/* Main Content Area */}
          <div className="p-4 flex-1 overflow-auto">

          {!moduleComplete ? (
            <>
              {/* Main Layout: Video + Learning Interface */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
                {/* Left: Video Feed + Feedback */}
                <div className="flex flex-col gap-3">
                  {/* Live Feed Label */}
                  <div className="bg-black/50 border-2 border-blue-400 rounded-lg px-4 py-2 backdrop-blur-sm">
                    <p className="font-pixel text-sm text-blue-300 text-center uppercase tracking-wider">
                      Live Feed
                    </p>
                  </div>

                  <VideoFeed
                    mode={videoFeedMode}
                    calibrationFrame={frame}
                    videoRef={videoRef}
                    canvasRef={canvasRef}
                    calibrationDone={calibrationDone}
                  />

                  {/* Incorrect Key Press Feedback */}
                  {lastTip && calibrationDone && (
                    <div className="bg-red-500/90 border-2 border-red-600 rounded-lg p-4 shadow-lg backdrop-blur-sm animate-in fade-in duration-300">
                      <div className="flex items-start gap-3">
                        <span className="text-2xl">Warning</span>
                        <div className="flex-1">
                          <p className="font-pixel text-xs text-white/80 uppercase tracking-wider mb-1">
                            Correction Needed
                          </p>
                          <p className="font-pixel text-sm text-white leading-relaxed">
                            {lastTip}
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Right: Learning Interface */}
                <div className="flex flex-col items-center justify-center gap-4">
                  {/* Module description */}
                  <PixelCard className="bg-black/50 border-2 border-yellow-400 backdrop-blur-sm px-6 py-3">
                    <p className="font-pixel text-xs text-yellow-200 text-center">
                      {MODULE_DESCRIPTIONS[moduleId]}
                    </p>
                  </PixelCard>

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
                    <PixelCard className="bg-black/50 border-2 border-purple-400 backdrop-blur-sm px-8 py-6">
                      <div className="text-center">
                        <p className="font-pixel text-[10px] text-white/70 uppercase tracking-widest mb-2">
                          Press this key
                        </p>
                        <p className="font-pixel text-6xl text-purple-400 animate-pulse drop-shadow-lg">
                          {targetChar}
                        </p>
                      </div>
                    </PixelCard>
                  )}

                  {/* Drill display with character-level feedback */}
                  {calibrationDone && currentDrill && (
                    <PixelCard className={`w-full max-w-md bg-black/60 border-2 border-yellow-300 backdrop-blur-sm ${
                      charFeedback[currentCharIndex] === "incorrect" ? "animate-shake" : ""
                    }`}>
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

                  {/* Progress & Metrics */}
                  {calibrationDone && (
                    <div className="flex gap-4">
                      <PixelCard variant="yellow" className="text-center px-4 py-2">
                        <p className="font-pixel text-[9px] text-foreground/60">Drill</p>
                        <p className="font-pixel text-lg text-foreground">{currentDrillIndex + 1}/{drills.length}</p>
                      </PixelCard>
                      <PixelCard variant="green" className="text-center px-4 py-2">
                        <p className="font-pixel text-[9px] text-foreground/60">Accuracy</p>
                        <p className="font-pixel text-lg text-foreground">{accuracyPercent}%</p>
                      </PixelCard>
                      <PixelCard variant="purple" className="text-center px-4 py-2">
                        <p className="font-pixel text-[9px] text-foreground/60">Correct</p>
                        <p className="font-pixel text-lg text-foreground">{correctCount}</p>
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
              <PixelCard className="p-8 text-center max-w-lg w-full bg-black/60 border-2 border-yellow-300 backdrop-blur-md shadow-2xl">
                <h2 className="font-pixel text-2xl text-yellow-400 mb-4">Module Complete!</h2>
                <p className="font-pixel text-sm text-white mb-2">
                  {MODULE_TITLES[moduleId]}
                </p>
                <p className="font-pixel text-xs text-white/80 mb-6">
                  {accuracyPercent >= 90
                    ? "Excellent work! You're mastering these keys!"
                    : accuracyPercent >= 70
                    ? "Good job! Keep practicing for even better accuracy."
                    : "Keep practicing! Focus on using the correct fingers."}
                </p>
                <div className="grid grid-cols-3 gap-4 mb-6">
                  <PixelCard className="bg-black/40 border border-yellow-400 text-center">
                    <p className="font-pixel text-[9px] text-yellow-400 mb-1">Accuracy</p>
                    <p className="font-pixel text-xl text-white">{accuracyPercent}%</p>
                  </PixelCard>
                  <PixelCard className="bg-black/40 border border-green-400 text-center">
                    <p className="font-pixel text-[9px] text-green-400 mb-1">Correct</p>
                    <p className="font-pixel text-xl text-white">{correctCount}</p>
                  </PixelCard>
                  <PixelCard className="bg-black/40 border border-red-400 text-center">
                    <p className="font-pixel text-[9px] text-red-400 mb-1">Incorrect</p>
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
