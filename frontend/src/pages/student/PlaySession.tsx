import { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { Logo } from "@/components/Logo";
import { PixelButton } from "@/components/PixelButton";
import { VideoFeed } from "@/components/VideoFeed";
import { TextPrompt } from "@/components/TextPrompt";
import { VirtualKeyboard } from "@/components/VirtualKeyboard";
import { MetricsPanel } from "@/components/MetricsPanel";
import { ErrorQueue } from "@/components/ErrorQueue";
import { CalibrationOverlay } from "@/components/CalibrationOverlay";
import { DetectionErrorOverlay } from "@/components/DetectionErrorOverlay";
import { SessionComplete } from "@/components/SessionComplete";
import { ArrowLeft } from "lucide-react";

const BASE_URL = import.meta.env.VITE_API_URL.replace("/api/auth", "");

declare global {
  interface Window {
    typedBuffer: string;
  }
}

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

// ============================================================
// Error Queue Entry
// ============================================================
interface ErrorQueueEntry {
  id: number;
  type: "incorrect_key" | "incorrect_finger";
  description: string;
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
  const [showCalibrationComplete, setShowCalibrationComplete] = useState(false);
  const [detectionError, setDetectionError] = useState<string | null>(null);
  const [frame, setFrame] = useState<string | null>(null);
  const [calibrationProgress, setCalibrationProgress] = useState({ detected: 0, required: 26 }); // ✅ 26-key model
  const calibrationDoneRef = useRef(false);
  const [correctCount, setCorrectCount] = useState(0);
  const [incorrectCount, setIncorrectCount] = useState(0);

  // ✅ REFS TO PREVENT CLOSURE ISSUES - Always have latest values
  const wordsRef = useRef<string[]>([]);
  const currentWordIndexRef = useRef(0);
  const userInputRef = useRef("");

  // ============================================================
  // Task 1: Calibration Timeout & Validation States
  // ============================================================
  const [calibrationError, setCalibrationError] = useState<string | null>(null);
  const calibrationStartTimeRef = useRef<number | null>(null);
  const calibrationTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ============================================================
  // Task 2: 10-Finger Monitoring States
  // ============================================================
  const [fingertipCount, setFingertipCount] = useState(0);
  const fingertipCountRef = useRef(0); // ✅ Ref to avoid closure trap
  const [fingerError, setFingerError] = useState(false);
  const fingerCheckTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const fingerBufferTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastFingerCheckRef = useRef<number>(Date.now());
  const lastGoodStateRef = useRef<number>(Date.now()); // ✅ Track when we last had >= 9 fingers
  const [typingBlocked, setTypingBlocked] = useState(false);
  const [lastKey, setLastKey] = useState<string | null>(null);
  const firstKeyTimeRef = useRef<number | null>(null);
  const endTimeRef = useRef<number | null>(null);
  const [completedExpected, setCompletedExpected] = useState(0);

  // ✅ Track last detection signature to prevent duplicate counting
  const lastEventRef = useRef<string | null>(null);

  // ✅ Additional metric states (Phase 1)
  const [fingerAccuracy, setFingerAccuracy] = useState(0);
  const [timingVariance, setTimingVariance] = useState(0);

  // 🧠 Track individual key errors
  const [errorHistory, setErrorHistory] = useState<
    { expected: string; pressed: string; tip: string }[]
  >([]);

  // ============================================================
  // SIMPLIFIED KEY COLORING: GREEN (fully correct) or RED (any error)
  // ============================================================
  // This function determines the visual feedback color for key presses.
  // - GREEN: Key press is FULLY CORRECT (right key + right finger)
  // - RED: Key press has ANY error (wrong key OR wrong finger)
  //
  // This simplified binary feedback makes it easier for students to
  // understand: green means perfect, red means something is wrong.
  // ============================================================
  const getKeyColor = (pressedKey: string, expectedKey: string, mlLabel: string) => {
    const correctKey = pressedKey === expectedKey;
    const correctFinger = mlLabel === "Correct";

    // Only green if BOTH key and finger are correct
    // Otherwise red for any kind of mistake
    if (correctKey && correctFinger) {
      return "green";  // Fully correct
    } else {
      return "red";    // Any error (wrong key OR wrong finger)
    }
  }

  // ============================================================
  // Timer State (30-second countdown)
  // ============================================================
  const TIMER_DURATION = 30;
  const [timeLeft, setTimeLeft] = useState(TIMER_DURATION);
  const timerStartedRef = useRef(false);
  const timerIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [sessionEnded, setSessionEnded] = useState(false);

  // ============================================================
  // Error Queue State (FIFO, max 5)
  // ============================================================
  const [errorQueue, setErrorQueue] = useState<ErrorQueueEntry[]>([]);
  const errorIdRef = useRef(0);
  // Guard: tracks the last key that already produced a queue error
  // so we never emit two errors (wrong key + wrong finger) for the same press
  const lastErroredKeyRef = useRef<string | null>(null);

  const pushError = useCallback((type: ErrorQueueEntry["type"], description: string, eventSignature?: string) => {
    // Use event signature (includes position + key) to prevent exact duplicates
    // This allows different errors for the same key at different positions
    if (eventSignature) {
      if (lastErroredKeyRef.current === eventSignature) return; // already logged
      lastErroredKeyRef.current = eventSignature;
    }
    const id = ++errorIdRef.current;
    setErrorQueue((prev) => {
      const next = [...prev, { id, type, description }];
      return next.length > 5 ? next.slice(next.length - 5) : next;
    });
  }, []);

  // ============================================================
  // Timer: start on first valid keypress, countdown to zero
  // ============================================================
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
      }, 1000);
    }
  }, [lastKey]);

  // Timer-driven session termination
  useEffect(() => {
    if (sessionEnded && detecting) {
      handleStopDetection();
    }
  }, [sessionEnded]);

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
    };
  }, []);

  // ============================================================
  // Task 3: Keyboard Shortcuts (Tab = Recalibrate, Enter = Finish)
  // ============================================================
  useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      // Priority 1: If calibration error popup is active, BLOCK all shortcuts
      if (calibrationError) {
        return;
      }

      // Priority 2: If finger error popup is active, ALLOW shortcuts
      // (user should be able to recalibrate or finish even with finger errors)

      if (e.key === "Tab") {
        e.preventDefault(); // Prevent default tab behavior (focus jumping)
        handleRecalibrate();
      } else if (e.key === "Enter") {
        e.preventDefault();
        handleFinish();
      }
    };

    window.addEventListener("keydown", handleGlobalKeyDown);
    return () => window.removeEventListener("keydown", handleGlobalKeyDown);
  }, [calibrationError, fingerError]); // Re-bind when popup states change

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

  // ============================================================
  // 🎯 FOCUS MANAGEMENT: Always keep hidden input focused
  // ============================================================
  useEffect(() => {
    const focusInput = () => {
      if (inputRef.current && !sessionEnded) {
        inputRef.current.focus();
      }
    };

    // Focus on mount
    focusInput();

    // Refocus on any click in the document
    const handleGlobalClick = () => {
      focusInput();
    };

    document.addEventListener("click", handleGlobalClick);

    return () => {
      document.removeEventListener("click", handleGlobalClick);
    };
  }, [sessionEnded]);

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

  // Auto-dismiss the "Calibration Complete" overlay after 2 seconds
  useEffect(() => {
    if (showCalibrationComplete) {
      const t = setTimeout(() => setShowCalibrationComplete(false), 2000);
      return () => clearTimeout(t);
    }
  }, [showCalibrationComplete]);

  // ============================================================
  // ✅ Sync refs (using useEffect for non-critical state)
  // ============================================================
  useEffect(() => {
    fingertipCountRef.current = fingertipCount;
  }, [fingertipCount]);

  useEffect(() => {
    wordsRef.current = words;
  }, [words]);

  // Note: currentWordIndex and userInput refs are updated SYNCHRONOUSLY
  // inside their setState calls for zero-latency metric updates

  // ════════════════════════════════════════════════════════════
  // ✅ 10-Finger Monitoring with HYSTERESIS (Prevents AI Flicker)
  // ════════════════════════════════════════════════════════════
  // HYSTERESIS: If the count was 10 and drops to 5 for < 1 second,
  // don't trigger the popup. The AI often flickers. Adding a grace
  // period prevents the UI from being annoying during split-second
  // AI detection errors.
  //
  // ✅ POST-SESSION GUARD: Stop all monitoring when session ends
  // ════════════════════════════════════════════════════════════
  useEffect(() => {
    // ✅ CRITICAL: Freeze monitoring after session ends to prevent ghost popups
    if (!calibrationDone || !detecting || sessionEnded) return;

    const now = Date.now();

    // ✅ LENIENT DISMISSAL: >= 9 fingers (MediaPipe often misses thumbs on home row)
    // This makes the system "human-friendly" while still enforcing proper hand position
    if (fingertipCount >= 9) {
      // Update last good state timestamp
      lastGoodStateRef.current = now;

      // Immediately clear error and unblock typing when hands are back
      setFingerError(false);
      setTypingBlocked(false);

      // Clear any pending buffer timeout
      if (fingerBufferTimeoutRef.current) {
        clearTimeout(fingerBufferTimeoutRef.current);
        fingerBufferTimeoutRef.current = null;
      }
      lastFingerCheckRef.current = now;
    }
    // If less than 9 fingers detected, apply hysteresis buffer
    else if (fingertipCount < 9) {
      // Clear any existing timeout
      if (fingerBufferTimeoutRef.current) {
        clearTimeout(fingerBufferTimeoutRef.current);
      }

      // ✅ HYSTERESIS: Add grace period if we were recently in good state
      // If we had 9+ fingers within the last 2 seconds, use longer buffer (2.5s)
      // This prevents transient AI drops from triggering the popup
      const timeSinceGoodState = now - lastGoodStateRef.current;
      const wasRecentlyGood = timeSinceGoodState < 2000; // Within 2 seconds
      const bufferDuration = wasRecentlyGood ? 2500 : 1500; // 2.5s if recently good, 1.5s otherwise

      fingerBufferTimeoutRef.current = setTimeout(() => {
        // ✅ FIX CLOSURE TRAP: Use Ref instead of state
        // This ensures we check the LATEST count, not the captured value
        if (fingertipCountRef.current < 9) {
          setFingerError(true);
          setTypingBlocked(true);
        }
      }, bufferDuration);
    }

    return () => {
      if (fingerBufferTimeoutRef.current) {
        clearTimeout(fingerBufferTimeoutRef.current);
      }
    };
  }, [fingertipCount, calibrationDone, detecting, sessionEnded]);

  // SSE handled in VideoFeed component
  useEffect(() => {
    if (!detecting) return;

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
            break;
          case "calibration_done":
            if (!calibrationDoneRef.current) {
              // Clear calibration timeout
              if (calibrationTimeoutRef.current) {
                clearTimeout(calibrationTimeoutRef.current);
                calibrationTimeoutRef.current = null;
              }

              // ✅ Validate that all 26 keys were detected (26-key model: a-z)
              const keysDetected = data.locked_keys || 0;
              if (keysDetected < 26) {
                setCalibrationError(
                  "Calibration incomplete - not all keys were detected. Please realign your keyboard properly."
                );
                setIsCalibrating(false);
                return;
              }

              // Calibration successful
              calibrationDoneRef.current = true;
              setCalibrationDone(true);
              setIsCalibrating(false);
              setShowCalibrationComplete(true);

              // Start 10-finger initial check (5 seconds after calibration)
              // ✅ Uses >= 9 threshold for human-friendly detection
              fingerCheckTimeoutRef.current = setTimeout(() => {
                if (fingertipCountRef.current < 9) {
                  setFingerError(true);
                  setTypingBlocked(true);
                }
              }, 5000);
            }
            break;
          case "error":
            setDetectionError(data.message);
            setDetecting(false);
            // Clear calibration timeout on error
            if (calibrationTimeoutRef.current) {
              clearTimeout(calibrationTimeoutRef.current);
              calibrationTimeoutRef.current = null;
            }
            break;
          case "frame":
            // Update frame and fingertip count from frame events
            setFrame(data.frame || null);
            if (typeof data.fingertip_count === 'number') {
              setFingertipCount(data.fingertip_count);
            }
            break;
          case "detection":
            // ============================================================
            // 🎯 EVENT-DRIVEN DETECTION (replaces polling)
            // ============================================================
            // Backend sends detection events via SSE when a key is pressed
            // This is the SINGLE SOURCE OF TRUTH for cursor advance,
            // metrics updates, and virtual keyboard coloring
            // ============================================================
            {
              // Update fingertip count from detection events
              if (typeof data.fingertip_count === 'number') {
                setFingertipCount(data.fingertip_count);
              }

              // Validate detection payload
              if (!data.key) return;

              const key = String(data.key).toUpperCase();
              // ✅ ALPHABET ONLY (A-Z) - ignore space and all other keys
              if (!/^[A-Z]$/.test(key)) return;

              // Block typing if finger error popup is active
              if (typingBlocked) return;

              const expectedKey = data.expected_key
                ? String(data.expected_key).toUpperCase()
                : key;

              const mlCorrect = data.ml_label === "Correct";
              const isCorrectFinal = mlCorrect && key === expectedKey;

              // ═══════════════════════════════════════════════════════════
              // ✅ ZERO-LATENCY FLAT ARCHITECTURE
              // ═══════════════════════════════════════════════════════════
              // • Read current state from REFS (always latest, no closures)
              // • Calculate position and validation FIRST
              // • Update metrics IMMEDIATELY at top level
              // • Update refs SYNCHRONOUSLY inside setState for instant sync
              // • NO NESTING - every setState is independent
              // ═══════════════════════════════════════════════════════════

              // ────────────────────────────────────────────────────────────
              // PHASE 1: READ CURRENT STATE FROM REFS
              // ────────────────────────────────────────────────────────────
              const currentWords = wordsRef.current;
              const currentWordIdx = currentWordIndexRef.current;
              const currentInput = userInputRef.current;

              // ────────────────────────────────────────────────────────────
              // PHASE 2: CALCULATE POSITION & VALIDATE
              // ────────────────────────────────────────────────────────────
              let globalCursorPos = 0;
              for (let i = 0; i < currentWordIdx; i++) {
                globalCursorPos += (currentWords[i]?.length || 0);
              }
              globalCursorPos += currentInput.length;

              // Duplicate prevention
              const signature = JSON.stringify({
                key,
                expectedKey,
                ml_label: data.ml_label,
                position: globalCursorPos,
              });
              if (lastEventRef.current === signature) return;
              lastEventRef.current = signature;

              // Validate we have more characters to type
              const fullText = currentWords.join("");
              const expectedChar = fullText[globalCursorPos];
              if (!expectedChar) return;

              const expectedWord = currentWords[currentWordIdx] || "";
              const nextCharInWord = expectedWord[currentInput.length];
              if (!nextCharInWord) return;

              // Calculate what the new state will be
              const newInput = currentInput + nextCharInWord;
              const newGlobalCursorPos = globalCursorPos + 1;
              const isWordComplete = newInput.length >= expectedWord.length;
              const isSessionComplete = newGlobalCursorPos >= fullText.length;

              // ────────────────────────────────────────────────────────────
              // PHASE 3: UPDATE METRICS FIRST (CRITICAL FOR UI RESPONSIVENESS)
              // ────────────────────────────────────────────────────────────
              const errorEventSignature = `${signature}-error`;

              if (isCorrectFinal) {
                setCorrectCount((prev) => {
                  const next = prev + 1;
                  console.log(`✅ Correct: ${prev} → ${next}`);
                  return next;
                });
              } else {
                setIncorrectCount((prev) => {
                  const next = prev + 1;
                  console.log(`❌ Incorrect: ${prev} → ${next}`);
                  return next;
                });

                // Determine error type
                const wrongKey = key !== expectedKey;
                const correctionTip = getCorrectionTip(expectedKey);

                if (wrongKey) {
                  pushError(
                    "incorrect_key",
                    `Wrong Key: Pressed "${key}" instead of "${expectedKey}"`,
                    errorEventSignature
                  );
                  setErrorHistory((prev) => [...prev, {
                    expected: expectedKey,
                    pressed: key,
                    tip: correctionTip
                  }]);
                } else {
                  pushError(
                    "incorrect_finger",
                    `Wrong Finger: Used incorrect finger for "${key}"`,
                    errorEventSignature
                  );
                  setErrorHistory((prev) => [...prev, {
                    expected: expectedKey,
                    pressed: key,
                    tip: correctionTip
                  }]);
                }
              }

              // ────────────────────────────────────────────────────────────
              // PHASE 4: VISUAL FEEDBACK (INDEPENDENT)
              // ────────────────────────────────────────────────────────────
              setActiveKeys((prev) => ({
                ...prev,
                [key]: getKeyColor(key, expectedKey, data.ml_label),
              }));

              setTimeout(() => {
                setActiveKeys((prev) => {
                  const updated = { ...prev };
                  delete updated[key];
                  return updated;
                });
              }, 300);

              setCharFeedback((prev) => ({
                ...prev,
                [globalCursorPos]: isCorrectFinal ? "correct" : "incorrect",
              }));

              // ────────────────────────────────────────────────────────────
              // PHASE 5: UPDATE INPUT & POSITION (WITH SYNCHRONOUS REF UPDATES)
              // ────────────────────────────────────────────────────────────

              // Update user input with SYNCHRONOUS ref update
              setUserInput(() => {
                const nextInput = isWordComplete ? "" : newInput;
                userInputRef.current = nextInput;
                return nextInput;
              });

              // Update word index with SYNCHRONOUS ref update
              if (isWordComplete) {
                setCurrentWordIndex((prev) => {
                  const next = prev + 1;
                  currentWordIndexRef.current = next;
                  return next;
                });
              }

              // Update last key
              setLastKey(key);

              // ────────────────────────────────────────────────────────────
              // PHASE 6: SESSION COMPLETION (IF APPLICABLE)
              // ────────────────────────────────────────────────────────────
              if (isSessionComplete) {
                console.log("🏁 Last character typed - ending session immediately");
                setSessionEnded(true);
                if (timerIntervalRef.current) {
                  clearInterval(timerIntervalRef.current);
                  timerIntervalRef.current = null;
                }
                endTimeRef.current = Date.now();
              }
            }
            break;
        }
      } catch (err) {
        console.error("SSE message parse error:", err);
      }
    };

    return () => source.close();
  }, [detecting]);

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

  // ============================================================
  // LOCAL KEYBOARD HANDLER — TEXT FEEDBACK ONLY
  // ============================================================
  // This handler provides immediate text-level feedback (green/red text)
  // but does NOT color the virtual keyboard keys.
  //
  // WHY: Virtual keyboard coloring is handled exclusively by the backend
  // detection polling loop (lines 494-555) which receives validated
  // results from the Python ML pipeline. This prevents the flicker issue
  // where keys would flash green (optimistic local check) then red
  // (actual backend validation).
  //
  // SINGLE SOURCE OF TRUTH: Backend validation determines key colors.
  // ============================================================
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    // Block backspace, delete, and all non-alphabet keys
    if (e.key === "Backspace" || e.key === "Delete") {
      e.preventDefault();
      return;
    }

    const pressedKey = e.key.toUpperCase();

    // ✅ ALPHABET ONLY (A-Z) - silently ignore space and all other keys
    if (!/^[A-Z]$/.test(pressedKey)) {
      e.preventDefault();
      return;
    }

    // The rest is handled by the SSE detection handler
    // This local handler is minimal and doesn't track errors
    // (SSE handler is the single source of truth)
  };

  // ============================================================
  // WPM + Accuracy Calculation
  // ============================================================
  useEffect(() => {
    // --------------------------
    // 1. START TIMER ON FIRST VALID PYTHON KEY
    // --------------------------
    if (
      firstKeyTimeRef.current === null &&
      lastKey !== null && /^[A-Za-z]$/.test(lastKey)
    ) {
      firstKeyTimeRef.current = Date.now();
    }

    // --------------------------
    // 2. COUNT EXPECTED WORD COMPLETION
    // --------------------------
    const expectedWord = words[completedExpected]?.toUpperCase() || "";

    if (!window.typedBuffer) window.typedBuffer = "";
    if (lastKey && /^[A-Za-z]$/.test(lastKey)) {
      window.typedBuffer += lastKey;
    }

    if (
      expectedWord.length > 0 &&
      window.typedBuffer.toUpperCase().endsWith(expectedWord)
    ) {
      setCompletedExpected((prev) => {
        const next = prev + 1;

        // 🔥 STOP TIMER PROPERLY WHEN LAST WORD IS DONE
        if (next >= words.length && endTimeRef.current === null) {
          endTimeRef.current = Date.now();
        }

        return next;
      });

      window.typedBuffer = "";
    }

    // --------------------------
    // 3. ACCURACY (finger-based)
    // --------------------------
    const totalFingerEvents = correctCount + incorrectCount;

    const accuracyVal =
      totalFingerEvents > 0
        ? Math.round((correctCount / totalFingerEvents) * 100)
        : 100;

    setAccuracy(accuracyVal);

    // --------------------------
    // 4. REAL-TIME WPM UPDATE
    // --------------------------
    const interval = setInterval(() => {
      if (firstKeyTimeRef.current !== null) {
        const now =
          endTimeRef.current !== null
            ? endTimeRef.current
            : Date.now();

        const minutesElapsed =
          (now - firstKeyTimeRef.current) / 1000 / 60;

        if (minutesElapsed > 0) {
          const wpmVal = Math.round(completedExpected / minutesElapsed);
          setWpm(wpmVal);
        }
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [lastKey, completedExpected, words, correctCount, incorrectCount]);

  // ============================================================
  // Detection Start/Stop
  // ============================================================
  const handleStartDetection = async () => {
    try {
      // ✅ IMMEDIATE STATE WIPE: Clear frame/progress FIRST
      // This ensures the browser immediately hides old green boxes
      setFrame(null);
      setCalibrationProgress({ detected: 0, required: 26 }); // ✅ 26-key model

      // Then reset all other state
      setDetectionError(null);
      setCalibrationError(null);
      setFingerError(false);
      setIsCalibrating(true);
      setCalibrationDone(false);
      setShowCalibrationComplete(false);
      calibrationDoneRef.current = false;
      setTypingBlocked(false);

      // Clear any existing timeouts
      if (calibrationTimeoutRef.current) {
        clearTimeout(calibrationTimeoutRef.current);
      }
      if (fingerCheckTimeoutRef.current) {
        clearTimeout(fingerCheckTimeoutRef.current);
      }
      if (fingerBufferTimeoutRef.current) {
        clearTimeout(fingerBufferTimeoutRef.current);
      }

      // Start 15-second calibration timeout
      calibrationStartTimeRef.current = Date.now();
      calibrationTimeoutRef.current = setTimeout(() => {
        if (!calibrationDoneRef.current) {
          setCalibrationError("No keyboard detected");
          setIsCalibrating(false);
        }
      }, 15000); // 15 seconds

      await startDetection();
      setDetecting(true);
    } catch (err) {
      console.error("Failed to start detection:", err);
      setIsCalibrating(false);
    }
  };

  // ============================================================
  // Recalibrate Function (replaces stop detection)
  // ============================================================
  const handleRecalibrate = async () => {
    try {
      // ✅ IMMEDIATE VISUAL WIPE: Clear frame/progress at the very start
      // This gives instant visual feedback that recalibration has started
      setFrame(null);
      setCalibrationProgress({ detected: 0, required: 26 }); // ✅ 26-key model

      // Stop existing detection
      if (detecting) {
        await stopDetection();
        setDetecting(false);
      }

      // Small delay to ensure clean shutdown
      await new Promise(resolve => setTimeout(resolve, 300));

      // Restart detection with fresh calibration
      await handleStartDetection();
    } catch (err) {
      console.error("Failed to recalibrate:", err);
    }
  };

  const handleStopDetection = async () => {
    try {
      await stopDetection();
      setDetecting(false);

      // 🔥 STOP WPM TIMER IMMEDIATELY
      if (endTimeRef.current === null) {
        endTimeRef.current = Date.now();
      }

      // 🔥 STOP all detection-based typing buffer logic
      window.typedBuffer = "";
    } catch (err) {
      console.error("Failed to stop detection:", err);
    }
  };

  useEffect(() => {
    const start = async () => await handleStartDetection();
    start();
    return () => {
      void handleStopDetection();
      // Cleanup all timeouts
      if (calibrationTimeoutRef.current) {
        clearTimeout(calibrationTimeoutRef.current);
      }
      if (fingerCheckTimeoutRef.current) {
        clearTimeout(fingerCheckTimeoutRef.current);
      }
      if (fingerBufferTimeoutRef.current) {
        clearTimeout(fingerBufferTimeoutRef.current);
      }
    };
  }, []);

  // ============================================================
  // DETECTION LOGIC MOVED TO SSE HANDLER (event-driven, no polling)
  // ============================================================

  // ============================================================
  // Finish Session (Phase 1 Metrics Foundation)
  // ============================================================
  const handleFinish = async () => {
    // Ensure detection is stopped before leaving the page
    if (detecting) {
      await handleStopDetection();
    }

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

  const isFinished = (currentWordIndex >= words.length && words.length > 0) || sessionEnded;

  // Stop detection AND timer when the typing session ends (either path)
  useEffect(() => {
    if (isFinished) {
      // Freeze the countdown — clear interval so no further ticks occur
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
        timerIntervalRef.current = null;
      }
      if (detecting) {
        handleStopDetection();
      }
    }
  }, [isFinished]);

  // ============================================================
  // UI — Compact Boxed Arena Layout
  // ============================================================
  return (
    <div className="min-h-screen flex items-start justify-center bg-background p-4 pt-6">
      <CalibrationOverlay
        isCalibrating={isCalibrating}
        showCalibrationComplete={showCalibrationComplete}
        calibrationProgress={calibrationProgress}
        frame={frame}
      />

      <DetectionErrorOverlay
        detectionError={detectionError}
        onRetry={() => {
          setDetectionError(null);
          handleStartDetection();
        }}
      />

      {/* ═══════════════════════════════════════════════════════
          CALIBRATION ERROR POPUP (Task 1)
          Blocks Tab/Enter shortcuts when active
          ═══════════════════════════════════════════════════════ */}
      {calibrationError && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="bg-card border-2 border-red-500 rounded-lg p-6 max-w-md shadow-2xl">
            <h2 className="text-xl font-pixel text-red-500 mb-4">
              ⚠️ Calibration Failed
            </h2>
            <p className="text-sm mb-6 text-foreground">
              {calibrationError}
            </p>
            <div className="flex gap-3 justify-center">
              <PixelButton
                variant="orange"
                onClick={() => {
                  setCalibrationError(null);
                  handleRecalibrate();
                }}
              >
                🔄 Recalibrate
              </PixelButton>
            </div>
          </div>
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════
          FINGER ERROR POPUP (Task 2)
          Allows Tab/Enter shortcuts when active
          Auto-dismisses when 9+ fingers detected (lenient for thumbs)
          ═══════════════════════════════════════════════════════ */}
      {fingerError && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <div className="bg-card border-2 border-yellow-500 rounded-lg p-6 max-w-md shadow-2xl">
            <h2 className="text-xl font-pixel text-yellow-500 mb-4">
              ✋ Hand Position Required
            </h2>
            <p className="text-sm mb-4 text-foreground">
              {calibrationDone && fingertipCount < 9
                ? "Improper resting position. Rest all fingers on home row keys (based on proper touch typing)."
                : "Rest your hands properly on the home row keys."}
            </p>
            <div className="text-center mb-4">
              <span className="text-2xl font-bold text-yellow-500">
                {fingertipCount}/10 fingers detected
              </span>
              <p className="text-xs text-muted-foreground mt-1">
                (9+ fingers required - lenient for thumbs)
              </p>
            </div>
            <p className="text-xs text-muted-foreground text-center">
              Typing is blocked until hands are properly positioned.
              <br />
              You can still use Tab (Recalibrate) or Enter (Finish Session).
            </p>
          </div>
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════
          OUTER BOXED CONTAINER — single enclosed arena
          ═══════════════════════════════════════════════════════ */}
      <div className="w-full max-w-[1200px] rounded-xl border border-border/50 shadow-lg bg-card/40 overflow-hidden flex flex-col">

        {/* ─── Header (inside the box) ─── */}
        <header className="flex items-center justify-between px-4 py-2 border-b border-border/30 bg-card/50 shrink-0">
          <div className="flex items-center gap-2">
            <PixelButton variant="secondary" size="sm" onClick={() => navigate("/student/play")}>
              <ArrowLeft size={14} />
            </PixelButton>
            <Logo />
          </div>
          <div className="font-pixel text-[9px] uppercase tracking-widest px-2.5 py-1 rounded-full border border-border/50 bg-card/60 text-muted-foreground">
            {sessionType === "evaluated" ? "🏆 Graded" : "🎮 Practice"} · Level {level}
          </div>
        </header>

        {/* ─── 3-Column Grid ─── */}
        <div className="grid grid-cols-[176px_1fr_200px] gap-3 p-3">

          <MetricsPanel
            correctCount={correctCount}
            incorrectCount={incorrectCount}
            wpm={wpm}
            accuracy={accuracy}
            timeLeft={timeLeft}
            timerDuration={TIMER_DURATION}
          />

          {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            CENTER — Camera → Text Prompt → Keyboard
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
          <section className="flex flex-col items-center gap-3 min-w-0">

            {!isFinished ? (
              <>
                <div>
                  <VideoFeed
                    detecting={detecting}
                    calibrationDone={calibrationDone}
                    baseUrl={BASE_URL}
                  />
                </div>

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

                <VirtualKeyboard activeKeys={activeKeys} />

                {/* Action Buttons */}
                <div className="flex gap-2 mt-1">
                  {!detecting ? (
                    <PixelButton variant="orange" size="sm" onClick={handleStartDetection}>
                      🎥 Start Detection
                    </PixelButton>
                  ) : (
                    <PixelButton variant="orange" size="sm" onClick={handleRecalibrate}>
                      🔄 Recalibrate (Tab)
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
                wpm={wpm}
                accuracy={accuracy}
                correctCount={correctCount}
                incorrectCount={incorrectCount}
                typedWordsLength={typedWords.length}
                errorHistory={errorHistory}
                onFinish={handleFinish}
              />
            )}
          </section>

          <ErrorQueue errorQueue={errorQueue} />
        </div>
      </div>
    </div>
  );
}
