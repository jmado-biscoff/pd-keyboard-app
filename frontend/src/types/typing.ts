/**
 * Centralized typing domain types
 * Used across PlaySession, SessionComplete, and displayBrain
 */

/**
 * Entry in the error history queue
 * Tracks details about each typing error for analysis
 */
export interface ErrorHistoryEntry {
  expected: string;
  pressed: string;
  tip: string;
  hand?: string;
  finger?: string;
}

/**
 * Entry in the session keystroke history
 * Records every keystroke with its status and metadata
 */
export interface SessionHistoryEntry {
  char: string;
  expected: string;
  status: "correct" | "wrong_key" | "wrong_finger" | "skipped";
  tip: string;
  hand?: string;
  finger?: string;
}

/**
 * Comprehensive session analysis from Display Brain
 * Contains all calculated metrics, grades, and feedback
 */
export interface SessionAnalysis {
  performanceSummary: string;
  masteryTip: string | null;
  isPerfect: boolean;
  totalErrors: number;
  mostFrequentKey: string | null;
  mostFrequentCount: number;
  netWpm: number;
  accuracyPercent: number;
  errorRate: number;
  letterGrade: string;
  compositeScore: number;
  completionRate: number;
}

/**
 * Session report for logging/debugging
 * Phase 1 metrics foundation
 */
export interface SessionReport {
  wpm: number;
  accuracy: number;
  correct_keystrokes: number;
  incorrect_keystrokes: number;
  fingerAccuracy: number;
  timingVariance: number;
  session_duration_sec: number;
}

/**
 * Error queue entry for real-time UI display
 */
export interface ErrorQueueEntry {
  id: number;
  type: "incorrect_key" | "incorrect_finger";
  description: string;
  pressedKey: string;
}

/**
 * Calibration state tracking
 */
export interface CalibrationState {
  done: boolean;
  detected: number;
  required: number;
  detected_keys: string[];
  frame?: string | null;
}
