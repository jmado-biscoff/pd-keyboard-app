import { useState, useEffect, useRef } from "react";

interface UseTypingStatsProps {
  lastKey: string | null;
  correctCountRef: React.MutableRefObject<number>;
  incorrectCountRef: React.MutableRefObject<number>;
  firstKeyTimeRef: React.MutableRefObject<number | null>;
  totalPausedTimeRef: React.MutableRefObject<number>;
  pauseStartTimeRef: React.MutableRefObject<number | null>;
  endTimeRef: React.MutableRefObject<number | null>;
  timerDuration: number;
  onSessionEnd: () => void;
}

export const useTypingStats = ({
  lastKey,
  correctCountRef,
  incorrectCountRef,
  firstKeyTimeRef,
  totalPausedTimeRef,
  pauseStartTimeRef,
  endTimeRef,
  timerDuration,
  onSessionEnd,
}: UseTypingStatsProps) => {
  const [timeLeft, setTimeLeft] = useState(timerDuration);
  const [wpm, setWpm] = useState(0);
  const [accuracy, setAccuracy] = useState(100);

  const timerStartedRef = useRef(false);
  const timerIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Timer: start on first valid keypress, countdown to zero
  useEffect(() => {
    if (timerStartedRef.current) return;
    if (lastKey !== null && /^[A-Za-z]$/.test(lastKey)) {
      timerStartedRef.current = true;
      timerIntervalRef.current = setInterval(() => {
        setTimeLeft((prev) => {
          if (prev <= 1) {
            clearInterval(timerIntervalRef.current!);
            timerIntervalRef.current = null;
            onSessionEnd();
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
  }, [lastKey, firstKeyTimeRef, correctCountRef, incorrectCountRef, totalPausedTimeRef, pauseStartTimeRef, endTimeRef, onSessionEnd]);

  // Calculate accuracy from counts
  useEffect(() => {
    const total = correctCountRef.current + incorrectCountRef.current;
    if (total > 0) {
      const accuracyVal = (correctCountRef.current / total) * 100;
      setAccuracy(accuracyVal);
    }
  }, [correctCountRef, incorrectCountRef]);

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
    };
  }, []);

  return {
    timeLeft,
    wpm,
    accuracy,
  };
};
