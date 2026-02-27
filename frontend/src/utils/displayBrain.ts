/**
 * Display Brain - Single Source of Truth for Typing Metrics
 *
 * This unified analysis function is used by:
 * - SessionComplete.tsx (UI display)
 * - PlaySession.tsx (database save)
 * - Play.tsx (dashboard display)
 *
 * Ensures 100% consistency across all metrics calculations.
 */

import type { ErrorHistoryEntry, SessionAnalysis } from "@/types/typing";

export type { ErrorHistoryEntry, SessionAnalysis };

/**
 * Analyzes a typing session and returns comprehensive performance metrics
 *
 * @param grossWpm - Gross WPM calculated as: (correct + incorrect) / (5 * minutesElapsed)
 * @param accuracy - Percentage accuracy (0-100), used only for isPerfect guard
 * @param correct - Number of fully-correct keystrokes (right key + right finger)
 * @param incorrect - Number of incorrect keystrokes (wrong key OR wrong finger)
 * @param errorHistory - Array of error details for analysis
 * @param correctKeysCount - Number of physically-correct key presses (right letter,
 *   regardless of finger technique). When provided, Accuracy is decoupled from
 *   technique so a user who hits the right key with the wrong finger scores 100%
 *   Accuracy but retains a high Error Rate.
 * @param totalExpectedChars - Total number of characters in the exercise. Used to
 *   calculate completion rate. If the user completes less than 50% of the exercise,
 *   the composite score is penalized proportionally.
 * @returns Complete session analysis with grades, scores, and insights
 */
export const analyzeSession = (
  grossWpm: number,
  accuracy: number,
  correct: number,
  incorrect: number,
  errorHistory: ErrorHistoryEntry[],
  correctKeysCount?: number,
  totalExpectedChars?: number
): SessionAnalysis => {
  const total = correct + incorrect;
  const isPerfect = errorHistory.length === 0 && accuracy === 100;

  // ═══════════════════════════════════════════════════════════
  // HIGH-PRECISION CALCULATIONS
  // ═══════════════════════════════════════════════════════════

  // 1. Calculate minutesElapsed from Gross WPM
  // Gross WPM = (correct + incorrect) / (5 * minutesElapsed)
  // Therefore: minutesElapsed = (correct + incorrect) / (5 * grossWpm)
  const minutesElapsed = grossWpm > 0 ? (correct + incorrect) / (5 * grossWpm) : 0;

  // 2. Net WPM = Gross WPM - (incorrectCount / (minutesElapsed * 5))
  const netWpm = minutesElapsed > 0
    ? grossWpm - (incorrect / (minutesElapsed * 5))
    : 0;

  // 3. Accuracy (%) — physical key accuracy, decoupled from finger technique.
  //    If correctKeysCount is provided, accuracy = correct letters / total keystrokes.
  //    This means hitting the right letter with the wrong finger scores 100% Accuracy
  //    while the technique penalty flows entirely through Error Rate (step 4).
  const accuracyPercent = total > 0
    ? ((correctKeysCount !== undefined ? correctKeysCount : correct) / total) * 100
    : 100;

  // 4. Error Rate (%) — includes BOTH wrong-key AND wrong-finger events.
  //    This is the technique penalty that lowers CompositeScore even when Accuracy is 100%.
  const errorRate = total > 0 ? (incorrect / total) * 100 : 0;

  // ═══════════════════════════════════════════════════════════
  // DYNAMIC BENCHMARKING (MaxWPM) - Refined Ranges
  // ═══════════════════════════════════════════════════════════
  let maxWpm = 30;
  if (netWpm > 19) {
    maxWpm = 30;
  } else if (netWpm > 14) {
    maxWpm = 19;
  } else if (netWpm > 7) {
    maxWpm = 14;
  } else {
    maxWpm = 7;
  }

  // ═══════════════════════════════════════════════════════════
  // COMPLETION FACTOR
  // No penalty if ≥50% of exercise completed; gradual penalty below 50%.
  // CompletionFactor = min(1, charactersTyped / (totalExpectedChars * 0.5))
  // ═══════════════════════════════════════════════════════════
  const completionRate = totalExpectedChars && totalExpectedChars > 0
    ? Math.min(1, total / totalExpectedChars)
    : (total > 0 ? 1 : 0);
  const completionFactor = totalExpectedChars && totalExpectedChars > 0
    ? Math.min(1, total / (totalExpectedChars * 0.5))
    : 1;

  // ═══════════════════════════════════════════════════════════
  // WEIGHTED COMPOSITE SCORE (with cap and completion factor)
  // ═══════════════════════════════════════════════════════════
  // RawScore = (Accuracy * 0.45) + ((100 - ErrorRate) * 0.30) + (min(1, NetWPM / MaxWPM) * 100 * 0.25)
  // CompositeScore = RawScore * CompletionFactor
  const rawCompositeScore =
    (accuracyPercent * 0.45) +
    ((100 - errorRate) * 0.30) +
    (Math.min(1, netWpm / maxWpm) * 100 * 0.25);
  const compositeScore = rawCompositeScore * completionFactor;

  // ═══════════════════════════════════════════════════════════
  // GRADE ASSIGNMENT (with Error-Dominant Penalty)
  // ═══════════════════════════════════════════════════════════
  let letterGrade = "D";

  // CRITICAL: Error-Dominant Penalty - overrides all other metrics
  if (errorRate >= 25) {
    letterGrade = "E";
  } else if (compositeScore >= 90) {
    letterGrade = "A";
  } else if (compositeScore >= 80) {
    letterGrade = "B";
  } else if (compositeScore >= 65) {
    letterGrade = "C";
  }

  // ═══════════════════════════════════════════════════════════
  // PERFORMANCE SUMMARY (based on Letter Grade)
  // ═══════════════════════════════════════════════════════════
  let performanceSummary = "";
  switch (letterGrade) {
    case "A":
      performanceSummary = "🏆 Excellent! Outstanding performance across speed, accuracy, and finger technique. You're mastering touch typing!";
      break;
    case "B":
      performanceSummary = "✅ Competent! Strong performance with good balance. Keep practicing to reach excellence.";
      break;
    case "C":
      performanceSummary = "⚠️ Developing. You're making progress! Focus on reducing errors and building consistent finger technique.";
      break;
    case "D":
      performanceSummary = "💡 Beginner. Take your time to build accuracy and proper finger positioning. Speed will come with practice.";
      break;
    case "E":
      performanceSummary = "🚩 Error-Dominant. Your error rate is too high (above 25%). Focus on using the correct fingers to build a reliable foundation before trying to type faster.";
      break;
  }

  // ═══════════════════════════════════════════════════════════
  // MASTERY TIP ANALYSIS
  // ═══════════════════════════════════════════════════════════
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
    netWpm,
    accuracyPercent,
    errorRate,
    letterGrade,
    compositeScore,
    completionRate: parseFloat((completionRate * 100).toFixed(1)),
  };
};

/**
 * Formats metrics for database storage (direct mapping from analysis)
 * Precision rules:
 * - Gross WPM: Integer (Math.round)
 * - All other metrics: 1 decimal place (handled by analyzeSession)
 */
export const formatMetricsForDatabase = (analysis: SessionAnalysis, grossWpm: number) => {
  return {
    wpm: Math.round(grossWpm),               // Integer (Gross WPM)
    netWpm: Math.round(analysis.netWpm),     // Integer (Net WPM)
    accuracy: parseFloat(analysis.accuracyPercent.toFixed(1)), // 1 decimal place
    errorRate: parseFloat(analysis.errorRate.toFixed(1)),    // 1 decimal place
    compositeScore: parseFloat(analysis.compositeScore.toFixed(1)), // 1 decimal place
    grade: analysis.letterGrade,
    completionRate: analysis.completionRate, // Already 1 decimal place
  };
};
