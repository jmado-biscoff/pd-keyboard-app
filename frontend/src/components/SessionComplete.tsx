import { useState, useEffect } from "react";
import { PixelCard } from "./PixelCard";
import { PixelButton } from "./PixelButton";
import { analyzeSession } from "@/utils/displayBrain";
import type { ErrorHistoryEntry, SessionHistoryEntry, SessionAnalysis } from "@/types/typing";

interface SessionCompleteProps {
  sessionEnded: boolean;
  wpm: number;
  accuracy: number;
  correctCount: number;
  incorrectCount: number;
  typedWordsLength: number;
  errorHistory: ErrorHistoryEntry[];
  sessionHistory: SessionHistoryEntry[];
  onFinish: () => void;
  finalAnalysis?: { analysis: SessionAnalysis; dbMetrics: any } | null;
}

export const SessionComplete = ({
  sessionEnded,
  wpm,
  accuracy,
  correctCount,
  incorrectCount,
  typedWordsLength,
  errorHistory,
  sessionHistory,
  onFinish,
  finalAnalysis,
}: SessionCompleteProps) => {
  // ════════════════════════════════════════════════════════════
  // LOADING STATE - Professional "Calculating Results..." Effect
  // ════════════════════════════════════════════════════════════
  const [isCalculating, setIsCalculating] = useState(true);

  // ════════════════════════════════════════════════════════════
  // SESSION REPLAY SLIDER STATE
  // ════════════════════════════════════════════════════════════
  const [replayIndex, setReplayIndex] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsCalculating(false);
    }, 500);

    return () => clearTimeout(timer);
  }, []);

  // Use finalAnalysis if available, otherwise calculate fallback
  const analysis = finalAnalysis?.analysis || analyzeSession(wpm, accuracy, correctCount, incorrectCount, errorHistory);
  const dbMetrics = finalAnalysis?.dbMetrics || {
    wpm,
    netWpm: analysis.netWpm,
    accuracy: analysis.accuracyPercent,
    errorRate: analysis.errorRate,
    compositeScore: analysis.compositeScore
  };

  // ════════════════════════════════════════════════════════════
  // TIMELINE SCROLLER LOGIC - Show window of 11 keys centered on active
  // ════════════════════════════════════════════════════════════
  const WINDOW_SIZE = 11;
  const centerOffset = Math.floor(WINDOW_SIZE / 2);

  const getVisibleEntries = () => {
    if (sessionHistory.length === 0) return [];

    const start = Math.max(0, replayIndex - centerOffset);
    const end = Math.min(sessionHistory.length, start + WINDOW_SIZE);
    const adjustedStart = Math.max(0, end - WINDOW_SIZE);

    return sessionHistory.slice(adjustedStart, end).map((entry, idx) => ({
      ...entry,
      originalIndex: adjustedStart + idx,
      isActive: adjustedStart + idx === replayIndex
    }));
  };

  const visibleEntries = getVisibleEntries();
  const currentEntry = sessionHistory[replayIndex];

  return (
    <PixelCard className="text-center py-8 shadow-xl w-full max-w-3xl mx-auto">
      {/* Header */}
      <p className="font-pixel text-2xl text-green-400 mb-2 drop-shadow-[0_0_12px_rgba(88,187,120,0.5)]">
        Session Complete!
      </p>
      <p className="font-pixel text-[10px] text-muted-foreground/70 uppercase tracking-widest mb-6">
        {sessionEnded ? "Time's up!" : "Level finished"}
      </p>

      {/* ═══════════════════════════════════════════════════════════════
          POSITION 1: 4-Column Metrics Panel
          ═══════════════════════════════════════════════════════════════ */}
      <div className="grid grid-cols-4 gap-3 max-w-md mx-auto mb-6">
        {/* Rating (Letter Grade) */}
        <div className="rounded-lg bg-purple-500/15 border border-purple-500/30 flex flex-col items-center justify-center py-4 min-h-[80px]">
          <p className="font-pixel text-[8px] text-purple-400/80 uppercase tracking-wider mb-1">Rating</p>
          <p className="font-pixel text-xl text-purple-400">
            {isCalculating ? "-" : analysis.letterGrade}
          </p>
        </div>

        {/* Score */}
        <div className="rounded-lg bg-green-500/15 border border-green-500/30 flex flex-col items-center justify-center py-4 min-h-[80px]">
          <p className="font-pixel text-[8px] text-green-400/80 uppercase tracking-wider mb-1">Score</p>
          <p className="font-pixel text-xl text-green-400">
            {isCalculating ? "-" : dbMetrics.compositeScore.toFixed(1)}
          </p>
        </div>

        {/* Net WPM */}
        <div className="rounded-lg bg-blue-500/15 border border-blue-500/30 flex flex-col items-center justify-center py-4 min-h-[80px]">
          <p className="font-pixel text-[8px] text-blue-400/80 uppercase tracking-wider mb-1">Net WPM</p>
          <p className="font-pixel text-xl text-blue-400">
            {isCalculating ? "-" : dbMetrics.netWpm}
          </p>
        </div>

        {/* Error Rate */}
        <div className="rounded-lg bg-red-500/15 border border-red-500/30 flex flex-col items-center justify-center py-4 min-h-[80px]">
          <p className="font-pixel text-[8px] text-red-400/80 uppercase tracking-wider mb-1">Error Rate</p>
          <p className="font-pixel text-xl text-red-400">
            {isCalculating ? "-" : `${dbMetrics.errorRate.toFixed(1)}%`}
          </p>
        </div>
      </div>

      {/* ═══════════════════════════════════════════════════════════════
          POSITION 2: Session Replay Section
          ═══════════════════════════════════════════════════════════════ */}
      {sessionHistory.length > 0 && (
        <div className="mb-6 max-w-2xl mx-auto">
          <p className="font-pixel text-[10px] uppercase tracking-widest text-muted-foreground/70 mb-3 text-left">
            🎬 Session Replay ({sessionHistory.length} keystrokes)
          </p>

          {/* Timeline Scroller - Horizontal sliding window with centered cursor */}
          <div className="bg-card/30 border border-border/50 rounded-lg p-4 mb-3 shadow-inner overflow-hidden">
            <div className="flex items-center justify-center gap-1 transition-transform duration-200">
              {visibleEntries.map((entry) => {
                const statusBgColor =
                  entry.status === "correct" ? "bg-green-500" :
                    entry.status === "wrong_finger" ? "bg-orange-500" :
                      entry.status === "skipped" ? "bg-black" : "bg-red-500";

                return (
                  <div
                    key={entry.originalIndex}
                    className={`transition-all duration-300 ${entry.isActive ? "scale-125 mx-2" : "scale-100 opacity-60"
                      }`}
                  >
                    {/* Character Box with Colored Background */}
                    <div
                      className={`font-pixel text-sm px-3 py-2 rounded ${statusBgColor} ${entry.isActive ? "border-2 border-white shadow-lg" : "border border-white/30"
                        } text-white transition-all duration-200`}
                    >
                      {entry.expected === " " ? "␣" : entry.expected}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Slider */}
          <div className="mb-4">
            <input
              type="range"
              min="0"
              max={sessionHistory.length - 1}
              value={replayIndex}
              onChange={(e) => setReplayIndex(parseInt(e.target.value))}
              className="w-full h-3 bg-card border-2 border-border/50 rounded-lg appearance-none cursor-pointer pixel-slider"
              style={{
                background: `linear-gradient(to right, rgba(34, 197, 94, 0.3) 0%, rgba(34, 197, 94, 0.3) ${(replayIndex / (sessionHistory.length - 1)) * 100}%, rgba(100, 116, 139, 0.2) ${(replayIndex / (sessionHistory.length - 1)) * 100}%, rgba(100, 116, 139, 0.2) 100%)`
              }}
            />
            <div className="flex justify-between mt-1">
              <span className="font-pixel text-[8px] text-muted-foreground">Keystroke: {replayIndex + 1}</span>
              <span className="font-pixel text-[8px] text-muted-foreground">Total: {sessionHistory.length}</span>
            </div>
          </div>

          {/* Replay Feedback Arena */}
          {currentEntry && (
            <div className="flex items-stretch gap-3">
              {/* Left: Pressed Key Box with Colored Background */}
              {(() => {
                const keyBgColor =
                  currentEntry.status === "correct" ? "bg-green-500" :
                    currentEntry.status === "wrong_finger" ? "bg-orange-500" :
                      currentEntry.status === "skipped" ? "bg-black" : "bg-red-500";

                return (
                  <div className={`flex-shrink-0 w-16 h-16 ${keyBgColor} border-2 border-white/30 rounded-lg flex items-center justify-center shadow-lg`}>
                    <span className="font-pixel text-2xl text-white uppercase">
                      {currentEntry.status === "skipped"
                        ? (currentEntry.expected === " " ? "␣" : currentEntry.expected)
                        : (currentEntry.char === " " ? "␣" : currentEntry.char)}
                    </span>
                  </div>
                );
              })()}

              {/* Right: Feedback Card */}
              {(() => {
                const isWrongFinger = currentEntry.status === "wrong_finger";
                const isWrongKey = currentEntry.status === "wrong_key";
                const isSkipped = currentEntry.status === "skipped";

                let bgColor = "bg-green-500";
                let icon = "✓";
                let title = "Correct!";
                let message = `Perfect! You pressed "${currentEntry.char}" with the correct finger.`;

                if (isSkipped) {
                  bgColor = "bg-black";
                  icon = "⏱";
                  title = "Skipped";
                  message = `You did not reach "${currentEntry.expected}" before time ran out.`;
                } else if (isWrongFinger) {
                  bgColor = "bg-orange-500";
                  icon = "⚠";
                  title = "Wrong Finger";
                  message = currentEntry.hand && currentEntry.finger
                    ? `You pressed '${currentEntry.char}' with ${currentEntry.hand} ${currentEntry.finger} finger.`
                    : `You pressed "${currentEntry.char}" but used the wrong finger.`;
                } else if (isWrongKey) {
                  bgColor = "bg-red-500";
                  icon = "✗";
                  title = "Wrong Key";
                  message = `You pressed "${currentEntry.char}" instead of "${currentEntry.expected}".`;
                }

                return (
                  <div className={`flex-1 ${bgColor} rounded-lg px-4 py-3 flex flex-col justify-center shadow-lg`}>
                    <div className="font-pixel text-[9px] text-white uppercase tracking-wider mb-1">
                      {icon} {title}
                    </div>
                    <p className="font-pixel text-[10px] text-white leading-tight mb-2">
                      {message}
                    </p>
                    {currentEntry.tip && (
                      <p className="font-pixel text-[9px] text-white/90 leading-tight">
                        {currentEntry.tip}
                      </p>
                    )}
                  </div>
                );
              })()}
            </div>
          )}
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════════════
          POSITION 3: Performance Summary
          ═══════════════════════════════════════════════════════════════ */}
      <div className="mb-4 max-w-2xl mx-auto">
        <p className="font-pixel text-[10px] uppercase tracking-widest text-muted-foreground/70 mb-2 text-left">
          📊 Performance Summary
        </p>
        <div className="font-pixel text-[11px] text-foreground text-left bg-card border border-border/50 rounded-lg px-4 py-3 leading-relaxed shadow-sm">
          {analysis.performanceSummary}
        </div>
      </div>

      {/* ═══════════════════════════════════════════════════════════════
          POSITION 4: Mastery Tip
          ═══════════════════════════════════════════════════════════════ */}
      {analysis.masteryTip && (
        <div className="mb-6 max-w-2xl mx-auto">
          <p className="font-pixel text-[10px] uppercase tracking-widest text-muted-foreground/70 mb-2 text-left">
            💡 Mastery Tip
          </p>
          <div
            className={`font-pixel text-[11px] text-foreground text-left border rounded-lg px-4 py-3 leading-relaxed shadow-sm ${analysis.isPerfect
              ? "bg-green-500/10 border-green-500/30"
              : "bg-blue-500/10 border-blue-500/30"
              }`}
          >
            {analysis.masteryTip}
          </div>
        </div>
      )}

      {/* Action Button */}
      <PixelButton variant="primary" size="md" className="mt-2" onClick={onFinish}>
        Back to Play
      </PixelButton>

      {/* Slider Styles */}
      <style>{`
        .pixel-slider::-webkit-slider-thumb {
          appearance: none;
          width: 16px;
          height: 16px;
          background: white;
          border: 2px solid rgba(34, 197, 94, 0.8);
          border-radius: 2px;
          cursor: pointer;
          box-shadow: 0 0 8px rgba(34, 197, 94, 0.5);
        }
        .pixel-slider::-moz-range-thumb {
          width: 16px;
          height: 16px;
          background: white;
          border: 2px solid rgba(34, 197, 94, 0.8);
          border-radius: 2px;
          cursor: pointer;
          box-shadow: 0 0 8px rgba(34, 197, 94, 0.5);
        }
        .pixel-slider::-webkit-slider-thumb:hover {
          box-shadow: 0 0 12px rgba(34, 197, 94, 0.8);
        }
        .pixel-slider::-moz-range-thumb:hover {
          box-shadow: 0 0 12px rgba(34, 197, 94, 0.8);
        }
      `}</style>
    </PixelCard>
  );
};
