import { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { Logo } from "@/components/Logo";
import { PixelButton } from "@/components/PixelButton";
import { VideoFeed } from "@/components/VideoFeed";
import { TextPrompt } from "@/components/TextPrompt";
import { VirtualKeyboard } from "@/components/VirtualKeyboard";
import { MetricsPanel } from "@/components/MetricsPanel";
import { ErrorQueue } from "@/components/ErrorQueue";
import { CompositeScoreLegend } from "@/components/CompositeScoreLegend";
import { CalibrationOverlay } from "@/components/CalibrationOverlay";
import { DetectionErrorOverlay } from "@/components/DetectionErrorOverlay";
import { SessionComplete } from "@/components/SessionComplete";
import { FingerErrorModal } from "@/components/session/FingerErrorModal";
import { ArrowLeft } from "lucide-react";
import { analyzeSession, formatMetricsForDatabase } from "@/utils/displayBrain";
import { getCorrectionTip, getKeyColor } from "@/utils/typingHelpers";
import { runCalibration } from "@/utils/calibrationClient";
import { initModels, startClientDetection, stopClientDetection, disposeAll } from "@/utils/detectionOrchestrator";
import { useAudio } from "@/contexts/AudioContext";
import type {
  ErrorHistoryEntry,
  SessionReport,
  ErrorQueueEntry,
  SessionHistoryEntry,
} from "@/types/typing";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:5000/api";

declare global {
  interface Window {
    typedBuffer: string;
  }
}

export default function PlaySession() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const sessionType = searchParams.get("type") || "practice";
  const level = parseInt(searchParams.get("level") || "1");
  const sessionId = searchParams.get("sessionId") || undefined;

  // Audio
  const { playCorrectSound, playErrorSound, muteMusicTemporarily } = useAudio();

  const [words, setWords] = useState<string[]>([]);
  const [typedWords, setTypedWords] = useState<string[]>([]);
  const [currentWordIndex, setCurrentWordIndex] = useState(0);
  const [userInput, setUserInput] = useState("");
  const [activeKeys, setActiveKeys] = useState<{ [key: string]: string }>({});
  const [charFeedback, setCharFeedback] = useState<{ [index: number]: string }>({});
  const [wpm, setWpm] = useState(0);
  const [accuracy, setAccuracy] = useState(100);
  const [detecting, setDetecting] = useState(false);
  const [isCalibrating, setIsCalibrating] = useState(false);
  const isCalibratingRef = useRef(false);
  const [calibrationDone, setCalibrationDone] = useState(false);
  const [showCalibrationComplete, setShowCalibrationComplete] = useState(false);
  const [detectionError, setDetectionError] = useState<string | null>(null);
  const [frame, setFrame] = useState<string | null>(null);
  const [calibrationProgress, setCalibrationProgress] = useState({ detected: 0, required: 26 });
  const [calibratedKeys, setCalibratedKeys] = useState<string[]>([]);
  const calibrationDoneRef = useRef(false);
  const [replayIndex, setReplayIndex] = useState(0);
  const [correctCount, setCorrectCount] = useState(0);
  const [incorrectCount, setIncorrectCount] = useState(0);
  const correctCountRef = useRef(0);
  const incorrectCountRef = useRef(0);
  // Physical key accuracy: increments when the correct letter is pressed regardless
  // of which finger was used. Decoupled from correctCount (key + finger both correct).
  const [correctKeysCount, setCorrectKeysCount] = useState(0);
  const correctKeysCountRef = useRef(0);

  // REFS TO PREVENT CLOSURE ISSUES
  const wordsRef = useRef<string[]>([]);
  const currentWordIndexRef = useRef(0);
  const userInputRef = useRef("");
  const aiPointerRef = useRef(0);

  // Calibration & detection refs
  const [calibrationError, setCalibrationError] = useState<string | null>(null);
  const calibrationStopRef = useRef<(() => void) | null>(null);
  const keyPositionsRef = useRef<Record<string, number[]>>({});

  // Video/Canvas refs for detection orchestrator
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // 10-Finger Monitoring States
  const [fingertipCount, setFingertipCount] = useState(0);
  const fingertipCountRef = useRef(0);
  const [leftFingersCount, setLeftFingersCount] = useState(0);
  const leftFingersCountRef = useRef(0);
  const [rightFingersCount, setRightFingersCount] = useState(0);
  const rightFingersCountRef = useRef(0);
  const [fingerError, setFingerError] = useState(false);
  const fingerErrorRef = useRef(false);
  const fingerCheckTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const fingerBufferTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastGoodStateRef = useRef<number>(Date.now());
  const [lastKey, setLastKey] = useState<string | null>(null);
  const firstKeyTimeRef = useRef<number | null>(null);
  const endTimeRef = useRef<number | null>(null);
  const [completedExpected, setCompletedExpected] = useState(0);

  const lastEventRef = useRef<string | null>(null);
  const totalPausedTimeRef = useRef<number>(0);
  const pauseStartTimeRef = useRef<number | null>(null);

  const [fingerAccuracy, setFingerAccuracy] = useState(0);
  const [timingVariance, setTimingVariance] = useState(0);

  const [errorHistory, setErrorHistory] = useState<ErrorHistoryEntry[]>([]);
  const errorHistoryRef = useRef<ErrorHistoryEntry[]>([]);

  const [sessionHistory, setSessionHistory] = useState<SessionHistoryEntry[]>([]);
  const sessionHistoryRef = useRef<SessionHistoryEntry[]>([]);

  // Timer State (30-second countdown)
  const TIMER_DURATION = 30;
  const [timeLeft, setTimeLeft] = useState(TIMER_DURATION);
  const timerStartedRef = useRef(false);
  const timerIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [sessionEnded, setSessionEnded] = useState(false);
  const [finalAnalysis, setFinalAnalysis] = useState<any>(null);

  // Error Queue State
  const [errorQueue, setErrorQueue] = useState<ErrorQueueEntry[]>([]);
  const errorIdRef = useRef(0);
  const lastErroredKeyRef = useRef<string | null>(null);

  // ML models initialization state
  const modelsInitializedRef = useRef(false);

  const pushError = useCallback((type: ErrorQueueEntry["type"], description: string, pressedKey: string, eventSignature?: string) => {
    if (eventSignature) {
      if (lastErroredKeyRef.current === eventSignature) return;
      lastErroredKeyRef.current = eventSignature;
    }
    const id = ++errorIdRef.current;
    setErrorQueue((prev) => {
      const next = [...prev, { id, type, description, pressedKey }];
      return next.length > 5 ? next.slice(next.length - 5) : next;
    });
  }, []);

  // Sync refs with state
  useEffect(() => { isCalibratingRef.current = isCalibrating; }, [isCalibrating]);
  useEffect(() => { fingerErrorRef.current = fingerError; }, [fingerError]);
  useEffect(() => { sessionHistoryRef.current = sessionHistory; }, [sessionHistory]);
  useEffect(() => { fingertipCountRef.current = fingertipCount; }, [fingertipCount]);
  useEffect(() => { wordsRef.current = words; }, [words]);

  // Timer: start on first valid keypress
  useEffect(() => {
    if (timerStartedRef.current) return;
    if (lastKey !== null && /^[A-Za-z]$/.test(lastKey)) {
      timerStartedRef.current = true;
      timerIntervalRef.current = setInterval(() => {
        setTimeLeft((prev) => {
          if (prev <= 1) {
            clearInterval(timerIntervalRef.current!);
            timerIntervalRef.current = null;
            setSessionEnded(true);
            return 0;
          }
          return prev - 1;
        });

        if (firstKeyTimeRef.current !== null) {
          const now = endTimeRef.current !== null
            ? endTimeRef.current
            : (pauseStartTimeRef.current ?? Date.now());
          const minutesElapsed = (now - firstKeyTimeRef.current - totalPausedTimeRef.current) / 1000 / 60;
          if (minutesElapsed > 0) {
            const grossWpm = Math.round((correctCountRef.current + incorrectCountRef.current) / (5 * minutesElapsed));
            setWpm(grossWpm);
          }
        }
      }, 1000);
    }
  }, [lastKey]);

  // Timer-driven session termination
  useEffect(() => {
    if (sessionEnded) {
      const fullText = wordsRef.current.join('').toUpperCase();
      const currentPosition = aiPointerRef.current;

      if (currentPosition < fullText.length) {
        const skippedEntries: SessionHistoryEntry[] = [];
        for (let i = currentPosition; i < fullText.length; i++) {
          skippedEntries.push({ char: '', expected: fullText[i], status: 'skipped', tip: 'Did not finish in time' });
        }
        if (skippedEntries.length > 0) {
          setSessionHistory(prev => [...prev, ...skippedEntries]);
        }
      }

      setFrame(null);
      setDetecting(false);
      if (detecting) handleStopDetection();
    }
  }, [sessionEnded]);

  // Cleanup timer on unmount
  useEffect(() => {
    return () => { if (timerIntervalRef.current) clearInterval(timerIntervalRef.current); };
  }, []);

  // Auto-dismiss fingerError when session ends
  useEffect(() => {
    if (sessionEnded || timeLeft === 0) {
      setFingerError(false);
      fingerErrorRef.current = false;
    }
  }, [sessionEnded, timeLeft]);

  // Keyboard Shortcuts (Tab = Recalibrate, Enter = Finish)
  useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      if (sessionEnded || (currentWordIndex >= words.length && words.length > 0)) return;
      if (calibrationError) return;

      if (e.key === "Tab") {
        e.preventDefault();
        handleRecalibrate();
      } else if (e.key === "Enter") {
        e.preventDefault();
        handleFinish();
      }
    };
    window.addEventListener("keydown", handleGlobalKeyDown);
    return () => window.removeEventListener("keydown", handleGlobalKeyDown);
  }, [calibrationError, fingerError, sessionEnded, currentWordIndex, words.length]);

  const inputRef = useRef<HTMLInputElement>(null);

  // Focus management
  useEffect(() => {
    const focusInput = () => { if (inputRef.current && !sessionEnded) inputRef.current.focus(); };
    focusInput();
    const handleGlobalClick = () => focusInput();
    document.addEventListener("click", handleGlobalClick);
    return () => document.removeEventListener("click", handleGlobalClick);
  }, [sessionEnded]);

  // Fetch typing data
  useEffect(() => {
    const fetchTypingData = async () => {
      try {
        const res = await fetch(`${API_BASE}/typing/level/${level}`);
        const data = await res.json();
        if (data && data.data) {
          const text = data.data.join(" ");
          const wordArray = text.split(" ");
          setWords(wordArray);
          aiPointerRef.current = 0;
        }
      } catch (error) {
        console.error("Error fetching typing data:", error);
      }
    };
    fetchTypingData();
  }, [level]);

  // Auto-dismiss calibration complete overlay
  useEffect(() => {
    if (showCalibrationComplete) {
      const t = setTimeout(() => setShowCalibrationComplete(false), 2000);
      return () => clearTimeout(t);
    }
  }, [showCalibrationComplete]);

  // 10-Finger Monitoring with hysteresis
  useEffect(() => {
    if (!calibrationDone || !detecting || sessionEnded) return;

    if (fingertipCount >= 9) {
      lastGoodStateRef.current = Date.now();
      setFingerError(false);
      if (fingerBufferTimeoutRef.current) {
        clearTimeout(fingerBufferTimeoutRef.current);
        fingerBufferTimeoutRef.current = null;
      }
    } else if (fingertipCount < 9) {
      if (fingerBufferTimeoutRef.current) clearTimeout(fingerBufferTimeoutRef.current);
      fingerBufferTimeoutRef.current = setTimeout(() => {
        if (fingertipCountRef.current < 9) setFingerError(true);
      }, 1000);
    }

    return () => {
      if (fingerBufferTimeoutRef.current) clearTimeout(fingerBufferTimeoutRef.current);
    };
  }, [fingertipCount, calibrationDone, detecting, sessionEnded]);

  // ============================================================
  // Handle detection events from orchestrator (replaces SSE)
  // ============================================================
  const handleDetectionEvent = useCallback((data: {
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
    // Update fingertip counts
    setFingertipCount(data.fingertip_count);
    fingertipCountRef.current = data.fingertip_count;
    setLeftFingersCount(data.left_fingers_count);
    leftFingersCountRef.current = data.left_fingers_count;
    setRightFingersCount(data.right_fingers_count);
    rightFingersCountRef.current = data.right_fingers_count;

    if (!data.key) return;
    const key = String(data.key).toUpperCase();
    if (!/^[A-Z]$/.test(key)) return;
    if (isCalibratingRef.current || fingerErrorRef.current) return;

    // Read current state from refs
    const currentWords = wordsRef.current;
    const currentWordIdx = currentWordIndexRef.current;
    const currentInput = userInputRef.current;
    const globalCursorPos = aiPointerRef.current;

    const fullText = currentWords.join("");
    const expectedChar = fullText[globalCursorPos];
    if (!expectedChar) return;

    const expectedKey = expectedChar.toUpperCase();
    const mlCorrect = data.ml_label === "Correct";
    const isCorrectFinal = mlCorrect && key === expectedKey;

    // Duplicate prevention
    const signature = JSON.stringify({ key, expectedKey, ml_label: data.ml_label, position: globalCursorPos });
    if (lastEventRef.current === signature) return;
    lastEventRef.current = signature;

    const newGlobalCursorPos = globalCursorPos + 1;
    const isSessionComplete = newGlobalCursorPos >= fullText.length;
    const expectedWord = currentWords[currentWordIdx] || "";
    const newInput = currentInput + expectedChar;
    const isWordComplete = newInput.length >= expectedWord.length;
    const errorEventSignature = `${signature}-error`;
    const detectedHand = data.hand ? String(data.hand) : undefined;
    const detectedFinger = data.finger ? String(data.finger) : undefined;

    // Track physical key accuracy: right letter pressed, regardless of finger technique.
    // This increments for both isCorrectFinal=true (right key + right finger) and
    // the wrong-finger case (right key + wrong finger), but NOT for wrong-key presses.
    if (key === expectedKey) {
      setCorrectKeysCount((prev) => { const next = prev + 1; correctKeysCountRef.current = next; return next; });
    }

    if (isCorrectFinal) {
      playCorrectSound();
      setCorrectCount((prev) => { const next = prev + 1; correctCountRef.current = next; return next; });
      setSessionHistory((prev) => [...prev, { char: key, expected: expectedKey, status: "correct", tip: "", hand: detectedHand, finger: detectedFinger }]);
    } else {
      playErrorSound();
      setIncorrectCount((prev) => { const next = prev + 1; incorrectCountRef.current = next; return next; });

      const wrongKey = key !== expectedKey;
      const correctionTip = getCorrectionTip(expectedKey);

      if (wrongKey) {
        pushError("incorrect_key", `Wrong Key: Pressed "${key}" instead of "${expectedKey}"`, key, errorEventSignature);
        setErrorHistory((prev) => { const next = [...prev, { expected: expectedKey, pressed: key, tip: correctionTip, hand: detectedHand, finger: detectedFinger }]; errorHistoryRef.current = next; return next; });
        setSessionHistory((prev) => [...prev, { char: key, expected: expectedKey, status: "wrong_key", tip: correctionTip, hand: detectedHand, finger: detectedFinger }]);
      } else {
        const fingerErrorMsg = detectedHand && detectedFinger
          ? `You pressed '${key}' with ${detectedHand} ${detectedFinger} finger.`
          : `Wrong Finger: Used incorrect finger for "${key}"`;
        pushError("incorrect_finger", fingerErrorMsg, key, errorEventSignature);
        setErrorHistory((prev) => { const next = [...prev, { expected: expectedKey, pressed: key, tip: correctionTip, hand: detectedHand, finger: detectedFinger }]; errorHistoryRef.current = next; return next; });
        setSessionHistory((prev) => [...prev, { char: key, expected: expectedKey, status: "wrong_finger", tip: correctionTip, hand: detectedHand, finger: detectedFinger }]);
      }
    }

    // Visual feedback
    setActiveKeys((prev) => ({ ...prev, [key]: getKeyColor(key, expectedKey, data.ml_label) }));
    setTimeout(() => { setActiveKeys((prev) => { const updated = { ...prev }; delete updated[key]; return updated; }); }, 300);
    setCharFeedback((prev) => ({ ...prev, [globalCursorPos]: isCorrectFinal ? "correct" : (key === expectedKey ? "wrong_finger" : "incorrect") }));

    // Update input & position
    setUserInput(() => { const nextInput = isWordComplete ? "" : newInput; userInputRef.current = nextInput; return nextInput; });
    if (isWordComplete) {
      setTypedWords((prev) => [...prev, newInput]);
      setCurrentWordIndex((prev) => { const next = prev + 1; currentWordIndexRef.current = next; return next; });
    }
    setLastKey(key);
    aiPointerRef.current += 1;

    // Session completion
    if (isSessionComplete) {
      endTimeRef.current = Date.now();
      if (firstKeyTimeRef.current !== null) {
        const finalMinutesElapsed = (endTimeRef.current - firstKeyTimeRef.current - totalPausedTimeRef.current) / 1000 / 60;
        if (finalMinutesElapsed > 0) {
          setWpm(Math.round((correctCountRef.current + incorrectCountRef.current) / (5 * finalMinutesElapsed)));
        }
      }
      if (timerIntervalRef.current) { clearInterval(timerIntervalRef.current); timerIntervalRef.current = null; }
      setSessionEnded(true);
    }
  }, [playCorrectSound, playErrorSound, pushError]);

  // Handle frame events from orchestrator
  const handleFrameEvent = useCallback((data: { fingertip_count: number; left_fingers_count: number; right_fingers_count: number }) => {
    setFingertipCount(data.fingertip_count);
    fingertipCountRef.current = data.fingertip_count;
    setLeftFingersCount(data.left_fingers_count);
    leftFingersCountRef.current = data.left_fingers_count;
    setRightFingersCount(data.right_fingers_count);
    rightFingersCountRef.current = data.right_fingers_count;
  }, []);

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
            onFrame: handleFrameEvent,
            onError: (err) => { setDetectionError(err.message); setDetecting(false); },
          },
          videoRef.current!,
          canvasRef.current!
        );
      } catch (err) {
        console.error("Failed to start client detection:", err);
        setDetectionError("Failed to start hand tracking. Please try again.");
      }
    };
    startDetect();
  }, [calibrationDone, detecting, handleDetectionEvent, handleFrameEvent]);

  // Typing Handlers
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (isCalibrating || fingerError) return;
    // When detection is active, the orchestrator handles cursor advancement
    // via handleDetectionEvent — don't also update from the input value
    if (detecting && calibrationDone) return;
    setUserInput(e.target.value);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (isCalibrating || fingerError) return;
    if (e.key === "Backspace" || e.key === "Delete") { e.preventDefault(); return; }
    const pressedKey = e.key.toUpperCase();
    if (!/^[A-Z]$/.test(pressedKey)) { e.preventDefault(); return; }
  };

  // WPM + Accuracy
  useEffect(() => {
    if (firstKeyTimeRef.current === null && lastKey !== null && /^[A-Za-z]$/.test(lastKey)) {
      firstKeyTimeRef.current = Date.now();
    }

    const expectedWord = words[completedExpected]?.toUpperCase() || "";
    if (!window.typedBuffer) window.typedBuffer = "";
    if (lastKey && /^[A-Za-z]$/.test(lastKey)) window.typedBuffer += lastKey;

    if (expectedWord.length > 0 && window.typedBuffer.toUpperCase().endsWith(expectedWord)) {
      setCompletedExpected((prev) => {
        const next = prev + 1;
        if (next >= words.length && endTimeRef.current === null) endTimeRef.current = Date.now();
        return next;
      });
      window.typedBuffer = "";
    }

    const totalFingerEvents = correctCountRef.current + incorrectCountRef.current;
    // Live accuracy display uses physical key accuracy (correct letters / total events)
    const accuracyVal = totalFingerEvents > 0 ? Math.round((correctKeysCountRef.current / totalFingerEvents) * 100) : 100;
    setAccuracy(accuracyVal);
  }, [lastKey, completedExpected, words]);

  // ============================================================
  // Start Detection (Hybrid: HTTP Calibration + Client Detection)
  // ============================================================
  const handleStartDetection = async () => {
    try {
      // Reset state
      setFrame(null);
      setCalibrationProgress({ detected: 0, required: 26 });
      setActiveKeys({});
      setCalibratedKeys([]);
      setDetectionError(null);
      setCalibrationError(null);
      setFingerError(false);
      setIsCalibrating(true);
      setCalibrationDone(false);
      setShowCalibrationComplete(false);
      calibrationDoneRef.current = false;

      if (fingerCheckTimeoutRef.current) clearTimeout(fingerCheckTimeoutRef.current);
      if (fingerBufferTimeoutRef.current) clearTimeout(fingerBufferTimeoutRef.current);

      // Reset AI pointer to current position
      const currentWords = wordsRef.current;
      const currentWordIdx = currentWordIndexRef.current;
      const currentInput = userInputRef.current;
      let globalCursorPos = 0;
      for (let i = 0; i < currentWordIdx; i++) {
        globalCursorPos += (currentWords[i]?.length || 0);
      }
      globalCursorPos += currentInput.length;
      aiPointerRef.current = globalCursorPos;

      // Initialize ML models (only once)
      if (!modelsInitializedRef.current) {
        try {
          await initModels();
          modelsInitializedRef.current = true;
        } catch (err) {
          console.error("Failed to initialize ML models:", err);
          setDetectionError("Failed to load ML models. Please refresh the page.");
          setIsCalibrating(false);
          return;
        }
      }

      // Create a temporary video element for calibration frame capture
      const calibVideoEl = document.createElement("video");
      calibVideoEl.autoplay = true;
      calibVideoEl.playsInline = true;
      calibVideoEl.muted = true;

      // Start HTTP-based calibration
      const { stop, stream } = await runCalibration(
        calibVideoEl,
        // onProgress
        (progress) => {
          setCalibrationProgress({ detected: progress.detected, required: progress.required });
          setCalibratedKeys(progress.detected_keys.map(k => k.toUpperCase()));
          if (progress.annotated_frame) setFrame(progress.annotated_frame);
        },
        // onComplete
        async (keyPositions) => {
          keyPositionsRef.current = keyPositions;
          calibrationDoneRef.current = true;
          setCalibrationDone(true);
          setIsCalibrating(false);
          setShowCalibrationComplete(true);
          setDetecting(true);

          // Stop calibration camera
          stream.getTracks().forEach(t => t.stop());
          calibVideoEl.srcObject = null;

          // Accumulate pause time
          if (firstKeyTimeRef.current && pauseStartTimeRef.current) {
            totalPausedTimeRef.current += (Date.now() - pauseStartTimeRef.current);
          }
          pauseStartTimeRef.current = null;

          // Client detection will be started by useEffect after VideoFeed re-renders with detection refs

          // Start 10-finger initial check
          fingerCheckTimeoutRef.current = setTimeout(() => {
            if (fingertipCountRef.current < 9) setFingerError(true);
          }, 5000);
        },
        // onError
        (msg) => {
          setCalibrationError(msg);
          setIsCalibrating(false);
          stream.getTracks().forEach(t => t.stop());
          calibVideoEl.srcObject = null;
        }
      );

      calibrationStopRef.current = stop;

    } catch (err) {
      console.error("Failed to start detection:", err);
      setIsCalibrating(false);
    }
  };

  // Recalibrate
  const handleRecalibrate = async () => {
    try {
      if (timerIntervalRef.current) { clearInterval(timerIntervalRef.current); timerIntervalRef.current = null; }
      timerStartedRef.current = false;
      pauseStartTimeRef.current = Date.now();
      setLastKey(null);
      setFingertipCount(0);
      fingertipCountRef.current = 0;

      setFrame(null);
      setCalibrationProgress({ detected: 0, required: 26 });
      setActiveKeys({});
      setCalibratedKeys([]);
      setCalibrationDone(false);
      setShowCalibrationComplete(false);
      setIsCalibrating(true);
      calibrationDoneRef.current = false;
      setDetecting(false);

      stopClientDetection();
      if (calibrationStopRef.current) { calibrationStopRef.current(); calibrationStopRef.current = null; }

      window.typedBuffer = "";
      await new Promise(resolve => setTimeout(resolve, 500));
      await handleStartDetection();
    } catch (err) {
      console.error("Failed to recalibrate:", err);
    }
  };

  const handleStopDetection = async () => {
    try {
      stopClientDetection();
      if (calibrationStopRef.current) { calibrationStopRef.current(); calibrationStopRef.current = null; }
      setDetecting(false);
      if (endTimeRef.current === null) endTimeRef.current = Date.now();
      window.typedBuffer = "";
    } catch (err) {
      console.error("Failed to stop detection:", err);
    }
  };

  // Auto-start on mount
  useEffect(() => {
    muteMusicTemporarily(true);
    const start = async () => await handleStartDetection();
    start();

    return () => {
      try {
        muteMusicTemporarily(false);
        disposeAll(); // Full cleanup: stop detection + dispose ML models on unmount
        if (calibrationStopRef.current) calibrationStopRef.current();
        if (fingerCheckTimeoutRef.current) clearTimeout(fingerCheckTimeoutRef.current);
        if (fingerBufferTimeoutRef.current) clearTimeout(fingerBufferTimeoutRef.current);
      } catch (err) {
        console.error("PlaySession cleanup error:", err);
      }
    };
  }, []);

  // Finish session
  const handleFinish = async () => {
    if (detecting) await handleStopDetection();
    navigate("/student/play");
  };

  // Calculate and save metrics
  const calculateAndSaveMetrics = async () => {
    if (!firstKeyTimeRef.current) return null;

    const endTime = endTimeRef.current || Date.now();
    const actualTypingDuration = endTime - firstKeyTimeRef.current - totalPausedTimeRef.current;
    const sessionDuration = Math.round(actualTypingDuration / 1000);
    const finalCorrectCount = correctCountRef.current;
    const finalIncorrectCount = incorrectCountRef.current;
    const finalCorrectKeysCount = correctKeysCountRef.current;
    const totalKeystrokes = finalCorrectCount + finalIncorrectCount;

    const minutesElapsed = sessionDuration / 60;
    const grossWpm = minutesElapsed > 0 ? (finalCorrectCount + finalIncorrectCount) / (5 * minutesElapsed) : 0;

    // Key accuracy % used for isPerfect guard; analyzeSession uses correctKeysCount
    // directly (6th param) for the decoupled accuracyPercent computation.
    const keyAccuracyPct = totalKeystrokes > 0 ? (finalCorrectKeysCount / totalKeystrokes) * 100 : 100;
    const totalExpectedChars = wordsRef.current.join(" ").length;
    const analysis = analyzeSession(
      grossWpm,
      keyAccuracyPct,
      finalCorrectCount,
      finalIncorrectCount,
      errorHistoryRef.current,
      finalCorrectKeysCount,
      totalExpectedChars
    );

    const dbMetrics = formatMetricsForDatabase(analysis, grossWpm);
    const wrongKeysCount = sessionHistoryRef.current.filter(entry => entry.status === "wrong_key").length;
    const wrongFingersCount = sessionHistoryRef.current.filter(entry => entry.status === "wrong_finger").length;
    const skippedCount = sessionHistoryRef.current.filter(entry => entry.status === "skipped").length;

    if (sessionType === "evaluated") {
      try {
        const userId = localStorage.getItem("userName") || "guest";
        await fetch(`${API_BASE}/results`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            userId, level,
            wpm: dbMetrics.wpm, accuracy: dbMetrics.accuracy, grade: dbMetrics.grade,
            sessionType, correctCount: finalCorrectCount,
            wrongKeysCount, wrongFingersCount, skippedCount,
            compositeScore: dbMetrics.compositeScore, netWpm: dbMetrics.netWpm, errorRate: dbMetrics.errorRate, completionRate: dbMetrics.completionRate,
            ...(sessionId ? { sessionId } : {}),
          }),
        });
      } catch (error) {
        console.error("Failed to auto-save result:", error);
      }
    }

    return { analysis, dbMetrics };
  };

  const isFinished = (currentWordIndex >= words.length && words.length > 0) || sessionEnded;

  // Stop detection when finished
  useEffect(() => {
    if (isFinished && !finalAnalysis) {
      setFrame(null);
      setDetecting(false);
      if (timerIntervalRef.current) { clearInterval(timerIntervalRef.current); timerIntervalRef.current = null; }
      if (detecting) handleStopDetection();

      (async () => {
        const result = await calculateAndSaveMetrics();
        if (result) setFinalAnalysis(result);
      })();
    }
  }, [isFinished, finalAnalysis]);

  // Determine VideoFeed mode
  const videoFeedMode = isCalibrating ? "calibration" : detecting ? "detection" : "idle";

  // ============================================================
  // UI
  // ============================================================
  return (
    <div className="min-h-screen flex items-start justify-center bg-[#D1BCDC] p-4 pt-6">
      <CalibrationOverlay
        isCalibrating={isCalibrating}
        showCalibrationComplete={showCalibrationComplete}
        calibrationProgress={calibrationProgress}
        calibratedKeys={calibratedKeys}
      />

      <DetectionErrorOverlay
        detectionError={detectionError}
        onRetry={() => { setDetectionError(null); handleStartDetection(); }}
      />

      {calibrationError && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="bg-card border-2 border-red-500 rounded-lg p-6 max-w-md shadow-2xl">
            <h2 className="text-xl font-pixel text-red-500 mb-4">Calibration Failed</h2>
            <p className="text-sm mb-6 text-foreground">{calibrationError}</p>
            <div className="flex gap-3 justify-center">
              <PixelButton variant="orange" onClick={() => { setCalibrationError(null); handleRecalibrate(); }}>
                Recalibrate
              </PixelButton>
            </div>
          </div>
        </div>
      )}

      <FingerErrorModal show={fingerError} leftFingersCount={leftFingersCount} rightFingersCount={rightFingersCount} />

      <div className="w-full max-w-[1200px] rounded-xl border border-border/50 shadow-lg bg-card/40 overflow-hidden flex flex-col">
        <header className="flex items-center justify-between px-4 py-2 border-b border-border/30 bg-card/50 shrink-0">
          <div className="flex items-center gap-2">
            <PixelButton variant="secondary" size="sm" onClick={() => navigate("/student/play")}>
              <ArrowLeft size={14} />
            </PixelButton>
            <Logo />
          </div>
          <div className="font-pixel text-[9px] uppercase tracking-widest px-2.5 py-1 rounded-full border border-border/50 bg-card/60 text-muted-foreground">
            {sessionType === "evaluated" ? "Activity / Graded" : "Practice"} · Level {level}
          </div>
        </header>

        <div className="grid grid-cols-[176px_1fr_200px] gap-3 p-3">
          <MetricsPanel
            correctCount={correctCount}
            incorrectCount={incorrectCount}
            correctKeysCount={correctKeysCount}
            wpm={wpm}
            accuracy={accuracy}
            timeLeft={timeLeft}
            timerDuration={TIMER_DURATION}
          />

          <section className="flex flex-col items-center gap-3 min-w-0">
            {!isFinished ? (
              <>
                <VideoFeed
                  mode={videoFeedMode}
                  calibrationFrame={frame}
                  videoRef={videoRef}
                  canvasRef={canvasRef}
                  calibrationDone={calibrationDone}
                />

                <div className="w-full max-w-2xl relative">
                  <TextPrompt
                    words={words}
                    currentWordIndex={currentWordIndex}
                    userInput={userInput}
                    charFeedback={charFeedback}
                    level={level}
                  />
                  <input
                    ref={inputRef}
                    type="text"
                    value={userInput}
                    onChange={handleChange}
                    onKeyDown={handleKeyDown}
                    className="absolute opacity-0 w-0 h-0 focus:outline-none"
                    autoFocus
                    aria-label="Typing input"
                  />
                </div>

                <VirtualKeyboard activeKeys={activeKeys} isCalibrating={isCalibrating} />

                <div className="flex gap-2 mt-1">
                  {!detecting && !isCalibrating ? (
                    <PixelButton variant="orange" size="sm" onClick={handleStartDetection}>
                      Start Detection
                    </PixelButton>
                  ) : (
                    <PixelButton variant="orange" size="sm" onClick={handleRecalibrate}>
                      Recalibrate (Tab)
                    </PixelButton>
                  )}
                  <PixelButton variant="primary" size="sm" onClick={handleFinish}>
                    Finish Session (Enter)
                  </PixelButton>
                </div>
              </>
            ) : (
              <SessionComplete
                sessionEnded={sessionEnded}
                wpm={finalAnalysis?.dbMetrics.wpm || wpm}
                accuracy={finalAnalysis?.dbMetrics.accuracy || accuracy}
                correctCount={correctCountRef.current}
                incorrectCount={incorrectCountRef.current}
                correctKeysCount={correctKeysCountRef.current}
                typedWordsLength={typedWords.length}
                errorHistory={errorHistory}
                sessionHistory={sessionHistoryRef.current}
                onFinish={handleFinish}
                finalAnalysis={finalAnalysis}
                replayIndex={replayIndex}
                onReplayIndexChange={setReplayIndex}
              />
            )}
          </section>

          {isFinished ? (
            <div className="flex flex-col gap-4">
              <CompositeScoreLegend currentGrade={finalAnalysis?.analysis?.letterGrade || ""} />
              {sessionHistoryRef.current.length > 0 && (() => {
                const entry = sessionHistoryRef.current[replayIndex];
                if (!entry) return null;
                return (
                  <div className="rounded-lg border border-border/40 bg-card/30 p-3 flex items-center justify-center gap-3">
                    <div className="flex flex-col items-center">
                      <p className="font-pixel text-[7px] text-muted-foreground/60 uppercase tracking-wider mb-1">Pressed</p>
                      {entry.status === "skipped" ? (
                        <div className="w-12 h-12 bg-black border-2 border-white/20 rounded-lg flex items-center justify-center shadow-lg">
                          <span className="font-pixel text-sm text-white/40">-</span>
                        </div>
                      ) : (
                        <div className={`w-12 h-12 border-2 rounded-lg flex items-center justify-center shadow-lg ${entry.status === "correct" ? "bg-green-500 border-green-600/50" :
                            entry.status === "wrong_finger" ? "bg-orange-500 border-orange-600/50" :
                              "bg-red-500 border-red-600/50"
                          }`}>
                          <span className="font-pixel text-xl text-white uppercase">
                            {entry.char === " " ? "\u2423" : entry.char}
                          </span>
                        </div>
                      )}
                    </div>
                    <span className="font-pixel text-sm text-muted-foreground/40">vs</span>
                    <div className="flex flex-col items-center">
                      <p className="font-pixel text-[7px] text-muted-foreground/60 uppercase tracking-wider mb-1">Expected</p>
                      <div className="w-12 h-12 bg-green-500 border-2 border-green-600/50 rounded-lg flex items-center justify-center shadow-lg">
                        <span className="font-pixel text-xl text-white uppercase">
                          {entry.expected === " " ? "\u2423" : entry.expected}
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })()}
              {/* ── Completion Rate Indicator ── */}
              {finalAnalysis && (
                <div className={`rounded-lg flex items-center justify-center py-3 ${finalAnalysis.analysis.completionRate < 50 ? "bg-red-500/80" : "bg-green-500/80"}`}>
                  <p className="font-pixel text-[9px] uppercase tracking-wider text-white mr-2">Completion:</p>
                  <p className="font-pixel text-sm text-white">
                    {`${finalAnalysis.analysis.completionRate.toFixed(1)}%`}
                  </p>
                  {finalAnalysis.analysis.completionRate < 50 && (
                    <span className="font-pixel text-[8px] text-white/80 ml-2">(penalty)</span>
                  )}
                </div>
              )}
            </div>
          ) : (
            <ErrorQueue errorQueue={errorQueue} />
          )}
        </div>
      </div>
    </div>
  );
}
