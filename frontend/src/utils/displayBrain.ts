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
 * @param accuracy - Percentage accuracy (0-100)
 * @param correct - Number of correct keystrokes
 * @param incorrect - Number of incorrect keystrokes
 * @param errorHistory - Array of error details for analysis
 * @returns Complete session analysis with grades, scores, and insights
 */
export const analyzeSession = (
  grossWpm: number,
  accuracy: number,
  correct: number,
  incorrect: number,
  errorHistory: ErrorHistoryEntry[]
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

  // 3. Accuracy (%)
  const accuracyPercent = total > 0 ? (correct / total) * 100 : 100;

  // 4. Error Rate (%)
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
  // WEIGHTED COMPOSITE SCORE (with cap to prevent scores > 100)
  // ═══════════════════════════════════════════════════════════
  // Score = (Accuracy * 0.45) + ((100 - ErrorRate) * 0.30) + (min(1, NetWPM / MaxWPM) * 100 * 0.25)
  const compositeScore =
    (accuracyPercent * 0.45) +
    ((100 - errorRate) * 0.30) +
    (Math.min(1, netWpm / maxWpm) * 100 * 0.25);

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
  };
};
