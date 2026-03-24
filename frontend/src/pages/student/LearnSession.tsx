import { useState, useEffect, useRef } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { Logo } from "@/components/Logo";
import { PixelButton } from "@/components/PixelButton";
import { PixelCard } from "@/components/PixelCard";
import { ArrowLeft } from "lucide-react";
import { CalibrationOverlay } from "@/components/CalibrationOverlay";
import { VideoFeed } from "@/components/VideoFeed";
import { KEYBOARD_IMAGES } from "@/utils/keyboardImages";
import { runCalibration } from "@/utils/calibrationClient";
import {
  initModels,
  startClientDetection,
  stopClientDetection,
  disposeAll,
} from "@/utils/detectionOrchestrator";
import { useAudio } from "@/contexts/AudioContext";
import bgVideo from "@/assets/bg2.mp4";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:5000/api";

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
  const [correctCount, setCorrectCount] = useState(0);
  const [incorrectCount, setIncorrectCount] = useState(0);
  // Physical key accuracy: correct letter pressed regardless of finger technique
  const [correctKeysCount, setCorrectKeysCount] = useState(0);
  const [lastError, setLastError] = useState<{ pressed: string; expected: string } | null>(null);
  const [lastPressedKey, setLastPressedKey] = useState<string | null>(null);
  const [isLastSuccess, setIsLastSuccess] = useState<boolean | null>(null);

  // Completion
  const [moduleComplete, setModuleComplete] = useState(false);

  // Refs for detection handler (avoid stale closures)
  const drillsRef = useRef<string[]>([]);
  const currentDrillIndexRef = useRef(0);
  const currentCharIndexRef = useRef(0);
  const correctCountRef = useRef(0);
  const incorrectCountRef = useRef(0);
  const correctKeysCountRef = useRef(0);
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
        const res = await fetch(`${API_BASE}/learn/module/${moduleId}`);
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
    setLastPressedKey(key);
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

    // Track for UI feedback
    setIsLastSuccess(isFullyCorrect);

    // Track physical key accuracy: right letter pressed regardless of finger technique.
    if (keyCorrect) {
      correctKeysCountRef.current++;
      setCorrectKeysCount(correctKeysCountRef.current);
    }

    if (isFullyCorrect) {
      correctCountRef.current++;
      setCorrectCount(correctCountRef.current);
      setLastError(null);
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
          setLastError(null);
        }
      } else {
        currentCharIndexRef.current = nextCharIndex;
        setCurrentCharIndex(nextCharIndex);
      }
    } else {
      incorrectCountRef.current++;
      setIncorrectCount(incorrectCountRef.current);
      playErrorSound();

      setLastError({ pressed: key, expected: expectedChar });

      setCharFeedback((prev) => ({ ...prev, [cIdx]: "incorrect" }));

      // Clear the dedup signature so the user can immediately retry the same
      // key at the same position without being blocked by the lock guard.
      lastEventRef.current = "";

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
            onFrame: () => { },
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
        await fetch(`${API_BASE}/student/learning-progress`, {
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
  const accuracyPercent = totalChars > 0 ? Math.round((correctKeysCount / totalChars) * 100) : 100;

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
              {/* ═══════════════════════════════════════════════════════════════
                  MIDDLE SECTION: Left Sidebar + Center Content + Right Sidebar
                  ═══════════════════════════════════════════════════════════════ */}
              <div className="flex gap-4">
                {/* Left: Vertical Metrics Stack */}
                {calibrationDone && (
                  <div className="flex flex-col gap-3 shrink-0">
                    <div className="bg-purple-500 border-2 border-purple-300 rounded-lg px-4 py-2 text-center">
                      <p className="font-pixel text-[9px] text-white/80">Drill</p>
                      <p className="font-pixel text-lg text-white">{currentDrillIndex + 1}/{drills.length}</p>
                    </div>
                    <div className="bg-red-500 border-2 border-red-300 rounded-lg px-4 py-2 text-center">
                      <p className="font-pixel text-[9px] text-white/80">Accuracy</p>
                      <p className="font-pixel text-lg text-white">{accuracyPercent}%</p>
                    </div>
                    <div className="bg-green-500 border-2 border-green-300 rounded-lg px-4 py-2 text-center">
                      <p className="font-pixel text-[9px] text-white/80">Correct</p>
                      <p className="font-pixel text-lg text-white">{correctCount}</p>
                    </div>
                  </div>
                )}

                {/* Center: VideoFeed + Press Key + Drill + Keyboard Image */}
                <div className="flex-1 flex flex-col gap-3">
                  {/* VideoFeed - whole visibility, slightly smaller max-width */}
                  <div className="flex justify-center rounded-lg">
                    <VideoFeed
                      mode={videoFeedMode}
                      calibrationFrame={frame}
                      videoRef={videoRef}
                      canvasRef={canvasRef}
                      calibrationDone={calibrationDone}
                    />
                  </div>

                  {/* Press Key + Drill side-by-side */}
                  {calibrationDone && (
                    <div className="flex gap-3">
                      {/* Current target letter */}
                      {targetChar && (
                        <div className="bg-black/50 border-2 border-purple-400 backdrop-blur-sm px-6 py-4 rounded-lg shrink-0">
                          <div className="text-center">
                            <p className="font-pixel text-[9px] text-white/70 uppercase tracking-widest mb-1">
                              Press this key
                            </p>
                            <p className="font-pixel text-5xl text-purple-400 animate-pulse drop-shadow-lg">
                              {targetChar}
                            </p>
                          </div>
                        </div>
                      )}

                      {/* Drill display with character-level feedback */}
                      {currentDrill && (
                        <div className={`flex-1 bg-black/60 border-2 border-yellow-300 backdrop-blur-sm rounded-lg min-h-[96px] flex items-center justify-center px-4 ${charFeedback[currentCharIndex] === "incorrect" ? "animate-shake" : ""}`}>
                          <div className="font-pixel text-xl text-center tracking-[0.4em]">
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
                        </div>
                      )}
                    </div>
                  )}

                  {/* Keyboard positioning image - fixed size to prevent layout shift */}
                  {keyboardImage && calibrationDone && (
                    <div className="flex justify-center w-full max-w-2xl mx-auto h-80">
                      <img
                        src={keyboardImage}
                        alt={`Press ${targetChar}`}
                        className="w-full h-full object-contain rounded-lg shadow-xl border-2 border-yellow-300/60"
                      />
                    </div>
                  )}
                </div>

                {/* Right: Correction Needed */}
                {calibrationDone && (
                  <div className="flex flex-col shrink-0">
                    <div className="bg-black/80 border-2 border-white/20 rounded-lg p-4 shadow-lg backdrop-blur-sm min-w-[140px]">
                      <div className="flex flex-col items-center gap-4">
                        <div className="flex flex-col items-center gap-1.5">
                          <div className={`w-14 h-14 border-2 rounded-lg flex items-center justify-center transition-all duration-200 ${
                            isLastSuccess === null ? "bg-white/5 border-white/20" :
                            isLastSuccess === true ? "bg-green-500/20 border-green-500 shadow-[0_0_10px_rgba(34,197,94,0.3)]" : 
                            "bg-red-500/20 border-red-500 shadow-[0_0_10px_rgba(239,68,68,0.3)]"
                          }`}>
                            <span className={`font-pixel text-2xl ${
                              isLastSuccess === null ? "text-white/20" :
                              isLastSuccess === true ? "text-green-400" : "text-red-400"
                            }`}>
                              {lastPressedKey || "-"}
                            </span>
                          </div>
                          <p className="font-pixel text-[8px] text-white/50 uppercase tracking-widest">Pressed</p>
                        </div>

                        <div className="w-8 h-[1px] bg-white/10" />

                        <div className="flex flex-col items-center gap-1.5">
                          <div className="w-14 h-14 bg-purple-500/15 border-2 border-purple-400 rounded-lg flex items-center justify-center shadow-[0_0_10px_rgba(192,132,252,0.2)]">
                            <span className="font-pixel text-2xl text-purple-400">{targetChar || "-"}</span>
                          </div>
                          <p className="font-pixel text-[8px] text-purple-300/70 uppercase tracking-widest">Expected</p>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Recalibrate button */}
              <div className="mt-4 text-center">
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
