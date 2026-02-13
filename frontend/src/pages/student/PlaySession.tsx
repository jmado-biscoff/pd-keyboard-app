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
import {
  getCorrectionTip,
  getKeyColor,
  startDetection,
  stopDetection,
  getDetectionStatus,
  setExpectedKeys,
} from "@/utils/typingHelpers";
import type {
  ErrorHistoryEntry,
  SessionReport,
  ErrorQueueEntry,
  SessionHistoryEntry,
} from "@/types/typing";

const BASE_URL = import.meta.env.VITE_API_URL.replace("/api/auth", "");

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
  const isCalibratingRef = useRef(false);
  const [calibrationDone, setCalibrationDone] = useState(false);
  const [showCalibrationComplete, setShowCalibrationComplete] = useState(false);
  const [detectionError, setDetectionError] = useState<string | null>(null);
  const [frame, setFrame] = useState<string | null>(null);
  const [calibrationProgress, setCalibrationProgress] = useState({ detected: 0, required: 26 }); // ✅ 26-key model
  const [calibratedKeys, setCalibratedKeys] = useState<string[]>([]);
  const calibrationDoneRef = useRef(false);
  const [replayIndex, setReplayIndex] = useState(0);
  const [correctCount, setCorrectCount] = useState(0);
  const [incorrectCount, setIncorrectCount] = useState(0);
  const correctCountRef = useRef(0);
  const incorrectCountRef = useRef(0);

  // ✅ REFS TO PREVENT CLOSURE ISSUES - Always have latest values
  const wordsRef = useRef<string[]>([]);
  const currentWordIndexRef = useRef(0);
  const userInputRef = useRef("");
  const aiPointerRef = useRef(0); // ✅ Independent AI pointer - tracks absolute index of NEXT character to grade

  // ============================================================
  // Task 1: Calibration Timeout & Validation States
  // ============================================================
  const [calibrationError, setCalibrationError] = useState<string | null>(null);
  const calibrationStartTimeRef = useRef<number | null>(null);
  const calibrationTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ============================================================
  // Task 2: 10-Finger Monitoring States (Dual-Hand Detection)
  // ============================================================
  const [fingertipCount, setFingertipCount] = useState(0);
  const fingertipCountRef = useRef(0); // ✅ Ref to avoid closure trap
  const [leftFingersCount, setLeftFingersCount] = useState(0);
  const leftFingersCountRef = useRef(0);
  const [rightFingersCount, setRightFingersCount] = useState(0);
  const rightFingersCountRef = useRef(0);
  const [fingerError, setFingerError] = useState(false);
  const fingerErrorRef = useRef(false); // ✅ Ref to avoid closure trap
  const fingerCheckTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const fingerBufferTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastFingerCheckRef = useRef<number>(Date.now());
  const lastGoodStateRef = useRef<number>(Date.now()); // ✅ Track when we last had >= 9 fingers
  const [lastKey, setLastKey] = useState<string | null>(null);
  const firstKeyTimeRef = useRef<number | null>(null);
  const endTimeRef = useRef<number | null>(null);
  const [completedExpected, setCompletedExpected] = useState(0);

  // ✅ Track last detection signature to prevent duplicate counting
  const lastEventRef = useRef<string | null>(null);
  const totalPausedTimeRef = useRef<number>(0);
  const pauseStartTimeRef = useRef<number | null>(null);

  // ✅ Additional metric states (Phase 1)
  const [fingerAccuracy, setFingerAccuracy] = useState(0);
  const [timingVariance, setTimingVariance] = useState(0);

  // 🧠 Track individual key errors
  const [errorHistory, setErrorHistory] = useState<ErrorHistoryEntry[]>([]);
  const errorHistoryRef = useRef<ErrorHistoryEntry[]>([]); // ✅ Ref to avoid stale closure

  // 🎬 Track complete session history for replay feature
  const [sessionHistory, setSessionHistory] = useState<SessionHistoryEntry[]>([]);
  const sessionHistoryRef = useRef<SessionHistoryEntry[]>([]);


  // ============================================================
  // Timer State (30-second countdown)
  // ============================================================
  const TIMER_DURATION = 30;
  const [timeLeft, setTimeLeft] = useState(TIMER_DURATION);
  const timerStartedRef = useRef(false);
  const timerIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [sessionEnded, setSessionEnded] = useState(false);
  const [finalAnalysis, setFinalAnalysis] = useState<any>(null); // Stores analysis for SessionComplete

  // ============================================================
  // Error Queue State (FIFO, max 5)
  // ============================================================
  const [errorQueue, setErrorQueue] = useState<ErrorQueueEntry[]>([]);
  const errorIdRef = useRef(0);
  // Guard: tracks the last key that already produced a queue error
  // so we never emit two errors (wrong key + wrong finger) for the same press
  const lastErroredKeyRef = useRef<string | null>(null);

  const pushError = useCallback((type: ErrorQueueEntry["type"], description: string, pressedKey: string, eventSignature?: string) => {
    // Use event signature (includes position + key) to prevent exact duplicates
    // This allows different errors for the same key at different positions
    if (eventSignature) {
      if (lastErroredKeyRef.current === eventSignature) return; // already logged
      lastErroredKeyRef.current = eventSignature;
    }
    const id = ++errorIdRef.current;
    setErrorQueue((prev) => {
      const next = [...prev, { id, type, description, pressedKey }];
      return next.length > 5 ? next.slice(next.length - 5) : next;
    });
  }, []);

  // ============================================================
  // Sync refs with state to avoid stale closures in SSE handler
  // ============================================================
  useEffect(() => {
    isCalibratingRef.current = isCalibrating;
  }, [isCalibrating]);

  useEffect(() => {
    fingerErrorRef.current = fingerError;
  }, [fingerError]);

  useEffect(() => {
    sessionHistoryRef.current = sessionHistory;
  }, [sessionHistory]);

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

        // Update Gross WPM every second
        if (firstKeyTimeRef.current !== null) {
          const now =
            endTimeRef.current !== null
              ? endTimeRef.current
              : (pauseStartTimeRef.current ?? Date.now());

          const minutesElapsed =
            (now - firstKeyTimeRef.current - totalPausedTimeRef.current) / 1000 / 60;

          if (minutesElapsed > 0) {
            const grossWpm = Math.round((correctCountRef.current + incorrectCountRef.current) / (5 * minutesElapsed));
            setWpm(grossWpm);
          }
        }
      }, 1000);
    }
  }, [lastKey]);

  // Timer-driven session termination - strict cleanup for privacy/security
  useEffect(() => {
    if (sessionEnded) {
      // Add skipped characters to session history
      const fullText = wordsRef.current.join('').toUpperCase();
      const currentPosition = aiPointerRef.current;

      if (currentPosition < fullText.length) {
        const skippedEntries: SessionHistoryEntry[] = [];
        for (let i = currentPosition; i < fullText.length; i++) {
          skippedEntries.push({
            char: '',
            expected: fullText[i],
            status: 'skipped',
            tip: 'Did not finish in time'
          });
        }

        if (skippedEntries.length > 0) {
          setSessionHistory(prev => [...prev, ...skippedEntries]);
        }
      }

      // Immediate cleanup: stop all frame capture and detection
      setFrame(null);
      setDetecting(false);
      if (detecting) {
        handleStopDetection();
      }
    }
  }, [sessionEnded]);

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
    };
  }, []);

  // ============================================================
  // Auto-dismiss fingerError when session ends or timer expires
  // ============================================================
  useEffect(() => {
    if (sessionEnded || timeLeft === 0) {
      setFingerError(false);
      fingerErrorRef.current = false;
    }
  }, [sessionEnded, timeLeft]);

  // ============================================================
  // Task 3: Keyboard Shortcuts (Tab = Recalibrate, Enter = Finish)
  // ============================================================
  useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      // Priority 0: Block all shortcuts on results screen
      if (sessionEnded || (currentWordIndex >= words.length && words.length > 0)) {
        return;
      }

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
  }, [calibrationError, fingerError, sessionEnded, currentWordIndex, words.length]); // Re-bind when popup states change


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
        const res = await fetch(`${BASE_URL}/api/typing/level/${level}`);
        const data = await res.json();
        if (data && data.data) {
          const text = data.data.join(" ");
          const wordArray = text.split(" ");
          setWords(wordArray);

          // Initialize AI pointer to start of text
          aiPointerRef.current = 0;

          await fetch(`${BASE_URL}/api/detect/set-expected`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              words: wordArray,
              startIndex: 0
            }),
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

      // Clear any pending buffer timeout
      if (fingerBufferTimeoutRef.current) {
        clearTimeout(fingerBufferTimeoutRef.current);
        fingerBufferTimeoutRef.current = null;
      }
      lastFingerCheckRef.current = now;
    }
    // If less than 9 fingers detected, apply strict 1-second buffer
    else if (fingertipCount < 9) {
      // Clear any existing timeout
      if (fingerBufferTimeoutRef.current) {
        clearTimeout(fingerBufferTimeoutRef.current);
      }

      // ✅ STRICT ENFORCEMENT: Fixed 1000ms (1 second) buffer
      // Popup appears exactly 1 second after fingers are removed
      const bufferDuration = 1000;

      fingerBufferTimeoutRef.current = setTimeout(() => {
        // ✅ FIX CLOSURE TRAP: Use Ref instead of state
        // This ensures we check the LATEST count, not the captured value
        if (fingertipCountRef.current < 9) {
          setFingerError(true);
        }
      }, bufferDuration);
    }

    return () => {
      if (fingerBufferTimeoutRef.current) {
        clearTimeout(fingerBufferTimeoutRef.current);
      }
    };
  }, [fingertipCount, calibrationDone, detecting, sessionEnded]);

  // SSE connection for real-time detection events
  useEffect(() => {
    if (!detecting) return;

    // Close SSE if session has ended (privacy/security)
    if (sessionEnded || isFinished) {
      return;
    }

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
          case "calibration_done":
            if (!calibrationDoneRef.current) {
              // ✅ ACCUMULATE PAUSE TIME
              if (firstKeyTimeRef.current && pauseStartTimeRef.current) {
                totalPausedTimeRef.current += (Date.now() - pauseStartTimeRef.current);
              }
              pauseStartTimeRef.current = null;

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
              fingertipCountRef.current = data.fingertip_count;
            }
            if (typeof data.left_fingers_count === 'number') {
              setLeftFingersCount(data.left_fingers_count);
              leftFingersCountRef.current = data.left_fingers_count;
            }
            if (typeof data.right_fingers_count === 'number') {
              setRightFingersCount(data.right_fingers_count);
              rightFingersCountRef.current = data.right_fingers_count;
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
                fingertipCountRef.current = data.fingertip_count;
              }
              if (typeof data.left_fingers_count === 'number') {
                setLeftFingersCount(data.left_fingers_count);
                leftFingersCountRef.current = data.left_fingers_count;
              }
              if (typeof data.right_fingers_count === 'number') {
                setRightFingersCount(data.right_fingers_count);
                rightFingersCountRef.current = data.right_fingers_count;
              }

              // Validate detection payload
              if (!data.key) return;

              const key = String(data.key).toUpperCase();
              // ✅ ALPHABET ONLY (A-Z) - ignore space and all other keys
              if (!/^[A-Z]$/.test(key)) return;

              // Block all input during calibration OR finger error
              // Recalibration = Full Pause (timer stops, no metrics)
              // Finger Error = Input Block Only (timer runs, WPM penalty)
              // Use refs to avoid stale closure issues in SSE handler
              if (isCalibratingRef.current || fingerErrorRef.current) return;

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
              // Use independent AI pointer - always points to the NEXT character to grade
              const globalCursorPos = aiPointerRef.current;

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

              // Calculate new state after this keystroke
              const newGlobalCursorPos = globalCursorPos + 1;
              const isSessionComplete = newGlobalCursorPos >= fullText.length;

              // Determine current word boundaries for word completion logic
              const expectedWord = currentWords[currentWordIdx] || "";
              const newInput = currentInput + expectedChar;
              const isWordComplete = newInput.length >= expectedWord.length;

              // ────────────────────────────────────────────────────────────
              // PHASE 3: UPDATE METRICS FIRST (CRITICAL FOR UI RESPONSIVENESS)
              // ────────────────────────────────────────────────────────────
              const errorEventSignature = `${signature}-error`;

              // Extract hand and finger data for tracking
              const detectedHand = data.hand ? String(data.hand) : undefined;
              const detectedFinger = data.finger ? String(data.finger) : undefined;

              if (isCorrectFinal) {
                setCorrectCount((prev) => {
                  const next = prev + 1;
                  correctCountRef.current = next;
                  console.log(`✅ Correct: ${prev} → ${next}`);
                  return next;
                });

                // Track correct keystroke in session history
                setSessionHistory((prev) => [...prev, {
                  char: key,
                  expected: expectedKey,
                  status: "correct",
                  tip: "",
                  hand: detectedHand,
                  finger: detectedFinger
                }]);
              } else {
                setIncorrectCount((prev) => {
                  const next = prev + 1;
                  incorrectCountRef.current = next;
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
                    key,
                    errorEventSignature
                  );
                  setErrorHistory((prev) => {
                    const next = [...prev, {
                      expected: expectedKey,
                      pressed: key,
                      tip: correctionTip,
                      hand: detectedHand,
                      finger: detectedFinger
                    }];
                    errorHistoryRef.current = next;
                    return next;
                  });

                  // Track wrong key in session history
                  setSessionHistory((prev) => [...prev, {
                    char: key,
                    expected: expectedKey,
                    status: "wrong_key",
                    tip: correctionTip,
                    hand: detectedHand,
                    finger: detectedFinger
                  }]);
                } else {
                  // Create specific finger error message
                  const fingerErrorMsg = detectedHand && detectedFinger
                    ? `You pressed '${key}' with ${detectedHand} ${detectedFinger} finger.`
                    : `Wrong Finger: Used incorrect finger for "${key}"`;

                  pushError(
                    "incorrect_finger",
                    fingerErrorMsg,
                    key,
                    errorEventSignature
                  );
                  setErrorHistory((prev) => {
                    const next = [...prev, {
                      expected: expectedKey,
                      pressed: key,
                      tip: correctionTip,
                      hand: detectedHand,
                      finger: detectedFinger
                    }];
                    errorHistoryRef.current = next;
                    return next;
                  });

                  // Track wrong finger in session history
                  setSessionHistory((prev) => [...prev, {
                    char: key,
                    expected: expectedKey,
                    status: "wrong_finger",
                    tip: correctionTip,
                    hand: detectedHand,
                    finger: detectedFinger
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
                [globalCursorPos]: isCorrectFinal
                  ? "correct"
                  : (key === expectedKey ? "wrong_finger" : "incorrect"),
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
                // Record completed word for session history
                setTypedWords((prev) => [...prev, newInput]);

                setCurrentWordIndex((prev) => {
                  const next = prev + 1;
                  currentWordIndexRef.current = next;
                  return next;
                });
              }

              // Update last key
              setLastKey(key);

              // Advance AI pointer to next character
              aiPointerRef.current += 1;

              // ────────────────────────────────────────────────────────────
              // PHASE 6: SESSION COMPLETION (IF APPLICABLE)
              // ────────────────────────────────────────────────────────────
              if (isSessionComplete) {
                console.log("🏁 Last character typed - ending session immediately");

                // Capture exact end time FIRST for high-precision calculation
                endTimeRef.current = Date.now();

                // Calculate final Gross WPM using precise timing
                if (firstKeyTimeRef.current !== null) {
                  const finalMinutesElapsed =
                    (endTimeRef.current - firstKeyTimeRef.current - totalPausedTimeRef.current) / 1000 / 60;

                  if (finalMinutesElapsed > 0) {
                    const finalGrossWpm = Math.round(
                      (correctCountRef.current + incorrectCountRef.current) / (5 * finalMinutesElapsed)
                    );
                    setWpm(finalGrossWpm);
                    console.log(`📊 Final Gross WPM: ${finalGrossWpm} (${finalMinutesElapsed.toFixed(2)} minutes)`);
                  }
                }

                // Stop timer interval
                if (timerIntervalRef.current) {
                  clearInterval(timerIntervalRef.current);
                  timerIntervalRef.current = null;
                }

                setSessionEnded(true);
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
    // Only update visual userInput - SSE handler manages word advancement
    if (isCalibrating || fingerError) return;
    setUserInput(e.target.value);
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
    // 🛑 PAUSE INPUT: Block typing if calibrating or hand error
    if (isCalibrating || fingerError) return;

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
    const totalFingerEvents = correctCountRef.current + incorrectCountRef.current;

    const accuracyVal =
      totalFingerEvents > 0
        ? Math.round((correctCountRef.current / totalFingerEvents) * 100)
        : 100;

    setAccuracy(accuracyVal);
  }, [lastKey, completedExpected, words]);

  // ============================================================
  // Detection Start/Stop
  // ============================================================
  const handleStartDetection = async () => {
    try {
      // ✅ IMMEDIATE STATE WIPE: Clear frame/progress FIRST
      // This ensures the browser immediately hides old green boxes
      setFrame(null);
      setCalibrationProgress({ detected: 0, required: 26 }); // ✅ 26-key model
      setActiveKeys({}); // Clear keyboard highlights during calibration
      setCalibratedKeys([]); // Reset calibrated keys list

      // Then reset all other state
      setDetectionError(null);
      setCalibrationError(null);
      setFingerError(false);
      setIsCalibrating(true);
      setCalibrationDone(false);
      setShowCalibrationComplete(false);
      calibrationDoneRef.current = false;

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

      // Calculate current global cursor position and send to backend
      const currentWords = wordsRef.current;
      const currentWordIdx = currentWordIndexRef.current;
      const currentInput = userInputRef.current;
      let globalCursorPos = 0;
      for (let i = 0; i < currentWordIdx; i++) {
        globalCursorPos += (currentWords[i]?.length || 0);
      }
      globalCursorPos += currentInput.length;

      // Reset AI pointer to current position (prevents skipping characters after recalibration)
      aiPointerRef.current = globalCursorPos;

      // Synchronize expected words with backend, including current position
      await fetch(`${BASE_URL}/api/detect/set-expected`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          words: currentWords,
          startIndex: globalCursorPos
        }),
      });

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
      // 🛑 STOP TIMER & RECORD PAUSE START
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
        timerIntervalRef.current = null;
      }
      timerStartedRef.current = false;
      pauseStartTimeRef.current = Date.now();
      setLastKey(null);
      setFingertipCount(0);
      fingertipCountRef.current = 0;

      // Instant UI feedback - wipe all calibration data immediately
      setFrame(null);
      setCalibrationProgress({ detected: 0, required: 26 });
      setActiveKeys({}); // Clear keyboard highlights during calibration
      setCalibratedKeys([]); // Reset calibrated keys list
      setCalibrationDone(false);
      setShowCalibrationComplete(false);
      setIsCalibrating(true);
      calibrationDoneRef.current = false;

      // Set detecting to false to prevent ghost detections
      setDetecting(false);

      // Force kill detection
      if (detecting) {
        await stopDetection();
      }

      // Wipe intermediate state
      window.typedBuffer = "";

      // Cool-down period - mandatory 1000ms delay for clean camera release
      await new Promise(resolve => setTimeout(resolve, 1000));

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

    // Navigate back to dashboard
    navigate("/student/play");
  };

  // ============================================================
  // Auto-Calculate and Save Metrics When Session Ends
  // ============================================================
  const calculateAndSaveMetrics = async () => {
    // ═══════════════════════════════════════════════════════════
    // TIMER FIX: Use firstKeyTimeRef (actual typing start) instead of startTime (page load)
    // ═══════════════════════════════════════════════════════════
    if (!firstKeyTimeRef.current) {
      console.warn("⚠️ No typing detected - cannot calculate metrics");
      return null;
    }

    const endTime = endTimeRef.current || Date.now();
    const actualTypingDuration = endTime - firstKeyTimeRef.current - totalPausedTimeRef.current;
    const sessionDuration = Math.round(actualTypingDuration / 1000);

    // ═══════════════════════════════════════════════════════════
    // USE REFS TO AVOID STALE STATE - Get final counts from refs
    // ═══════════════════════════════════════════════════════════
    const finalCorrectCount = correctCountRef.current;
    const finalIncorrectCount = incorrectCountRef.current;
    const totalKeystrokes = finalCorrectCount + finalIncorrectCount;

    // ═══════════════════════════════════════════════════════════
    // SINGLE SOURCE OF TRUTH: Shared Display Brain Analysis
    // ═══════════════════════════════════════════════════════════

    // 1. Calculate Gross WPM from actual typing timer (excludes calibration)
    const minutesElapsed = sessionDuration / 60;
    const grossWpm = minutesElapsed > 0
      ? (finalCorrectCount + finalIncorrectCount) / (5 * minutesElapsed)
      : 0;

    // 2. Use the tracked error history ref for high-precision analysis
    const errorHistory = errorHistoryRef.current;

    // 3. Call the EXACT SAME analyzeSession function that SessionComplete.tsx uses
    const analysis = analyzeSession(
      grossWpm,
      totalKeystrokes > 0 ? (finalCorrectCount / totalKeystrokes) * 100 : 100,
      finalCorrectCount,
      finalIncorrectCount,
      errorHistory
    );

    // 4. Format metrics with 2-decimal precision for database
    const dbMetrics = formatMetricsForDatabase(analysis, grossWpm);

    // 5. Calculate detailed error breakdown from session history (using ref for accuracy)
    const wrongKeysCount = sessionHistoryRef.current.filter(entry => entry.status === "wrong_key").length;
    const wrongFingersCount = sessionHistoryRef.current.filter(entry => entry.status === "wrong_finger").length;
    const skippedCount = sessionHistoryRef.current.filter(entry => entry.status === "skipped").length;

    const report: SessionReport = {
      wpm: dbMetrics.wpm,
      accuracy: dbMetrics.accuracy,
      correct_keystrokes: finalCorrectCount,
      incorrect_keystrokes: finalIncorrectCount,
      fingerAccuracy,
      timingVariance,
      session_duration_sec: sessionDuration,
    };

    console.log("📊 Session Report:", report);
    console.log("📊 Display Brain Analysis (100% Unified):", {
      correctCount: finalCorrectCount,
      incorrectCount: finalIncorrectCount,
      wrongKeysCount,
      wrongFingersCount,
      skippedCount,
      ...dbMetrics,
      letterGrade: analysis.letterGrade,
      performanceSummary: analysis.performanceSummary,
    });

    // ═══════════════════════════════════════════════════════════
    // AUTO-SAVE: Immediately save to database when session ends
    // ═══════════════════════════════════════════════════════════
    if (sessionType === "evaluated") {
      try {
        const userId = localStorage.getItem("userName") || "guest";
        await fetch(`${BASE_URL}/api/results`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            userId,
            level,
            wpm: dbMetrics.wpm,
            accuracy: dbMetrics.accuracy,
            grade: dbMetrics.grade,
            sessionType,
            correctCount: finalCorrectCount,
            wrongKeysCount,
            wrongFingersCount,
            skippedCount,
            compositeScore: dbMetrics.compositeScore,
            netWpm: dbMetrics.netWpm,
            errorRate: dbMetrics.errorRate,
            ...(sessionId ? { sessionId } : {}),
          }),
        });
        console.log("✅ Auto-saved to MongoDB immediately on session end");
      } catch (error) {
        console.error("❌ Failed to auto-save result:", error);
      }
    }

    // Return analysis object for SessionComplete to display
    return { analysis, dbMetrics };
  };

  const isFinished = (currentWordIndex >= words.length && words.length > 0) || sessionEnded;

  // Stop detection AND timer when the typing session ends (either path)
  useEffect(() => {
    if (isFinished && !finalAnalysis) {
      // Strict session-end cleanup for privacy/security
      setFrame(null);
      setDetecting(false);

      // Freeze the countdown
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
        timerIntervalRef.current = null;
      }

      // Stop Python camera process
      if (detecting) {
        handleStopDetection();
      }

      // ═══════════════════════════════════════════════════════════
      // AUTO-SAVE: Calculate metrics and save to database immediately
      // ═══════════════════════════════════════════════════════════
      (async () => {
        const result = await calculateAndSaveMetrics();
        if (result) {
          setFinalAnalysis(result);
          console.log("🎯 Final analysis calculated and auto-saved:", result);
        }
      })();
    }
  }, [isFinished, finalAnalysis]);

  // ============================================================
  // UI — Compact Boxed Arena Layout
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
          FINGER ERROR POPUP (Dual-Hand Detection)
          Allows Tab/Enter shortcuts when active
          Auto-dismisses when session ends or timer expires
          ═══════════════════════════════════════════════════════ */}
      <FingerErrorModal
        show={fingerError}
        leftFingersCount={leftFingersCount}
        rightFingersCount={rightFingersCount}
      />

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
            {sessionType === "evaluated" ? "🏆 Activity / Graded" : "🎮 Practice"} · Level {level}
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
                <VideoFeed
                  detecting={detecting}
                  calibrationDone={calibrationDone}
                  baseUrl={BASE_URL}
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
                wpm={finalAnalysis?.dbMetrics.wpm || wpm}
                accuracy={finalAnalysis?.dbMetrics.accuracy || accuracy}
                correctCount={correctCountRef.current}
                incorrectCount={incorrectCountRef.current}
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

              {/* Pressed vs Expected — driven by replay slider */}
              {sessionHistoryRef.current.length > 0 && (() => {
                const entry = sessionHistoryRef.current[replayIndex];
                if (!entry) return null;
                return (
                  <div className="rounded-lg border border-border/40 bg-card/30 p-3 flex items-center justify-center gap-3">
                    {/* Pressed Key */}
                    <div className="flex flex-col items-center">
                      <p className="font-pixel text-[7px] text-muted-foreground/60 uppercase tracking-wider mb-1">Pressed</p>
                      {entry.status === "skipped" ? (
                        <div className="w-12 h-12 bg-black border-2 border-white/20 rounded-lg flex items-center justify-center shadow-lg">
                          <span className="font-pixel text-sm text-white/40">-</span>
                        </div>
                      ) : (
                        <div className={`w-12 h-12 border-2 rounded-lg flex items-center justify-center shadow-lg ${
                          entry.status === "correct" ? "bg-green-500 border-green-600/50" :
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

                    {/* Expected Key */}
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
            </div>
          ) : (
            <ErrorQueue errorQueue={errorQueue} />
          )}
        </div>
      </div>
    </div>
  );
}
