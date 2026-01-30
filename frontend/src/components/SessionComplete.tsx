import { PixelCard } from "./PixelCard";
import { PixelButton } from "./PixelButton";

interface ErrorHistoryEntry {
  expected: string;
  pressed: string;
  tip: string;
}

interface SessionCompleteProps {
  sessionEnded: boolean;
  wpm: number;
  accuracy: number;
  correctCount: number;
  incorrectCount: number;
  typedWordsLength: number;
  errorHistory: ErrorHistoryEntry[];
  onFinish: () => void;
}

// Analyze error patterns and generate mastery tip
interface SessionAnalysis {
  performanceSummary: string;
  masteryTip: string | null;
  isPerfect: boolean;
  totalErrors: number;
  mostFrequentKey: string | null;
  mostFrequentCount: number;
}

const analyzeSession = (
  wpm: number,
  accuracy: number,
  correct: number,
  incorrect: number,
  errorHistory: ErrorHistoryEntry[]
): SessionAnalysis => {
  const total = correct + incorrect;
  const errorRate = total > 0 ? (incorrect / total) * 100 : 0;
  const isPerfect = errorHistory.length === 0 && accuracy === 100;

  // 1. Performance Summary
  let performanceSummary = "";
  if (accuracy >= 95 && wpm >= 30) {
    performanceSummary = "🏆 Excellent! High accuracy and good speed. You're mastering touch typing!";
  } else if (accuracy >= 85) {
    performanceSummary = "✅ Good job! You're typing accurately. Keep practicing to build speed.";
  } else if (accuracy >= 70) {
    performanceSummary = "⚠️ Focus on accuracy. Slow down and ensure each keystroke is correct.";
  } else {
    performanceSummary = "💡 Accuracy needs improvement. Review finger positions on the home row.";
  }

  // 2. Analyze errors for Mastery Tip
  let masteryTip: string | null = null;
  let mostFrequentKey: string | null = null;
  let mostFrequentCount = 0;

  if (isPerfect) {
    masteryTip = "🌟 Perfect Technique! Keep maintaining this level of accuracy and focus.";
  } else if (errorHistory.length > 0) {
    // Count frequency of missed keys
    const missedKeys: Record<string, number> = {};
    errorHistory.forEach((err) => {
      missedKeys[err.expected] = (missedKeys[err.expected] || 0) + 1;
    });

    // Find most frequently missed key
    const topMissed = Object.entries(missedKeys)
      .sort((a, b) => b[1] - a[1])[0];

    if (topMissed) {
      mostFrequentKey = topMissed[0];
      mostFrequentCount = topMissed[1];

      const tipForKey = errorHistory.find((e) => e.expected === mostFrequentKey)?.tip;

      // Determine if it's mostly wrong key or wrong finger
      const errorsForKey = errorHistory.filter((e) => e.expected === mostFrequentKey);
      const wrongKeyCount = errorsForKey.filter((e) => e.pressed !== e.expected).length;
      const wrongFingerCount = errorsForKey.length - wrongKeyCount;

      // Calculate potential improvement
      const potentialImprovement = Math.round((mostFrequentCount / total) * 100);

      if (wrongKeyCount > wrongFingerCount) {
        // Wrong key pressed
        masteryTip = `Focus on finding the '${mostFrequentKey}' key accurately to improve accuracy by ${potentialImprovement}%. ${tipForKey || ""}`;
      } else {
        // Wrong finger used
        masteryTip = `${tipForKey || `Practice using the correct finger for '${mostFrequentKey}'.`} This could improve your accuracy by ${potentialImprovement}%.`;
      }
    }
  }

  return {
    performanceSummary,
    masteryTip,
    isPerfect,
    totalErrors: errorHistory.length,
    mostFrequentKey,
    mostFrequentCount,
  };
};

export const SessionComplete = ({
  sessionEnded,
  wpm,
  accuracy,
  correctCount,
  incorrectCount,
  typedWordsLength,
  errorHistory,
  onFinish,
}: SessionCompleteProps) => {
  const analysis = analyzeSession(wpm, accuracy, correctCount, incorrectCount, errorHistory);

  return (
    <PixelCard className="text-center py-8 shadow-xl w-full max-w-3xl mx-auto">
      {/* Header */}
      <p className="font-pixel text-2xl text-green-400 mb-2 drop-shadow-[0_0_12px_rgba(88,187,120,0.5)]">
        🎉 Session Complete!
      </p>
      <p className="font-pixel text-[10px] text-muted-foreground/70 uppercase tracking-widest mb-6">
        {sessionEnded ? "⏰ Time's up!" : "✅ Level finished"}
      </p>

      {/* Metrics Grid */}
      <div className="grid grid-cols-4 gap-3 max-w-md mx-auto mb-6">
        <div className="rounded-lg bg-yellow-500/15 border border-yellow-500/30 px-3 py-2 text-center">
          <p className="font-pixel text-[8px] text-yellow-400/80 uppercase tracking-wider mb-1">WPM</p>
          <p className="font-pixel text-xl text-yellow-400">{wpm}</p>
        </div>
        <div className="rounded-lg bg-green-500/15 border border-green-500/30 px-3 py-2 text-center">
          <p className="font-pixel text-[8px] text-green-400/80 uppercase tracking-wider mb-1">Accuracy</p>
          <p className="font-pixel text-xl text-green-400">{accuracy}%</p>
        </div>
        <div className="rounded-lg bg-blue-500/15 border border-blue-500/30 px-3 py-2 text-center">
          <p className="font-pixel text-[8px] text-blue-400/80 uppercase tracking-wider mb-1">Correct</p>
          <p className="font-pixel text-xl text-blue-400">{correctCount}</p>
        </div>
        <div className="rounded-lg bg-red-500/15 border border-red-500/30 px-3 py-2 text-center">
          <p className="font-pixel text-[8px] text-red-400/80 uppercase tracking-wider mb-1">Errors</p>
          <p className="font-pixel text-xl text-red-400">{incorrectCount}</p>
        </div>
      </div>

      {/* Performance Summary */}
      <div className="mb-4 max-w-2xl mx-auto">
        <p className="font-pixel text-[10px] uppercase tracking-widest text-muted-foreground/70 mb-2 text-left">
          📊 Performance Summary
        </p>
        <div className="font-pixel text-[11px] text-foreground text-left bg-card border border-border/50 rounded-lg px-4 py-3 leading-relaxed shadow-sm">
          {analysis.performanceSummary}
        </div>
      </div>

      {/* Mastery Tip */}
      {analysis.masteryTip && (
        <div className="mb-6 max-w-2xl mx-auto">
          <p className="font-pixel text-[10px] uppercase tracking-widest text-muted-foreground/70 mb-2 text-left">
            💡 Mastery Tip
          </p>
          <div
            className={`font-pixel text-[11px] text-foreground text-left border rounded-lg px-4 py-3 leading-relaxed shadow-sm ${
              analysis.isPerfect
                ? "bg-green-500/10 border-green-500/30"
                : "bg-blue-500/10 border-blue-500/30"
            }`}
          >
            {analysis.masteryTip}
          </div>
        </div>
      )}

      {/* Review Arena - Scrollable Error List */}
      {errorHistory.length > 0 && (
        <div className="mb-6 max-w-2xl mx-auto">
          <p className="font-pixel text-[10px] uppercase tracking-widest text-muted-foreground/70 mb-2 text-left">
            📝 Review Arena ({analysis.totalErrors} {analysis.totalErrors === 1 ? "mistake" : "mistakes"})
          </p>
          <div
            className="max-h-[300px] overflow-y-auto bg-card/30 border border-border/50 rounded-lg p-3 shadow-inner"
            style={{
              scrollbarWidth: 'thin',
              scrollbarColor: 'rgba(156, 163, 175, 0.3) rgba(0, 0, 0, 0.1)'
            }}
          >
            <div className="flex flex-col gap-2">
              {errorHistory.map((error, idx) => {
                const isWrongKey = error.pressed !== error.expected;
                const errorType = isWrongKey ? "Wrong Key Pressed" : "Wrong Finger";

                return (
                  <PixelCard
                    key={idx}
                    className="p-3 bg-card border-red-500/30 shadow-sm"
                  >
                    {/* Error Header */}
                    <div className="flex items-center gap-2 mb-2">
                      <span className="font-pixel text-[8px] uppercase tracking-wider px-2 py-1 rounded bg-red-500/20 text-red-300">
                        {errorType}
                      </span>
                      {isWrongKey && (
                        <span className="font-pixel text-[9px] text-muted-foreground">
                          for <span className="text-green-400 font-bold">"{error.expected}"</span>
                        </span>
                      )}
                    </div>

                    {/* Error Body */}
                    <p className="font-pixel text-[10px] text-foreground leading-relaxed">
                      You pressed{" "}
                      <span className="text-red-400 font-bold">"{error.pressed}"</span>
                      {" "}instead of{" "}
                      <span className="text-green-400 font-bold">"{error.expected}"</span>.
                    </p>

                    {/* Correction Tip */}
                    <p className="font-pixel text-[9px] text-blue-400 mt-2 leading-relaxed">
                      ✓ {error.tip}
                    </p>
                  </PixelCard>
                );
              })}
            </div>
          </div>
          <style>{`
            .max-h-\\[300px\\]::-webkit-scrollbar {
              width: 8px;
            }
            .max-h-\\[300px\\]::-webkit-scrollbar-track {
              background: rgba(0, 0, 0, 0.1);
              border-radius: 4px;
            }
            .max-h-\\[300px\\]::-webkit-scrollbar-thumb {
              background: rgba(156, 163, 175, 0.3);
              border-radius: 4px;
            }
            .max-h-\\[300px\\]::-webkit-scrollbar-thumb:hover {
              background: rgba(156, 163, 175, 0.5);
            }
          `}</style>
        </div>
      )}

      {/* Action Button */}
      <PixelButton variant="primary" size="md" className="mt-2" onClick={onFinish}>
        🏠 Back to Play
      </PixelButton>
    </PixelCard>
  );
};
