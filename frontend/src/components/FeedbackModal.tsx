// src/components/FeedbackModal.tsx
import { motion } from "framer-motion";
import { PixelCard } from "@/components/PixelCard";
import { PixelButton } from "@/components/PixelButton";

export interface SessionReport {
  accuracy: number;
  wpm: number;
  correct_keystrokes: number;
  incorrect_keystrokes: number;
  finger_errors: Record<string, number>;
  timing_variance: number;
  streaks: number;
  session_duration_sec: number;
}

export default function FeedbackModal({
  report,
  onClose,
}: {
  report: SessionReport;
  onClose: () => void;
}) {
  const totalMistakes = Object.values(report.finger_errors).reduce(
    (a, b) => a + b,
    0
  );
  const mostErrorFinger = Object.entries(report.finger_errors).sort(
    (a, b) => b[1] - a[1]
  )[0]?.[0];

  // 🧠 Adaptive feedback messages
  const feedbackMessages: string[] = [];

  if (report.accuracy < 85)
    feedbackMessages.push(
      "Focus on accuracy before speed — steady typing helps long-term retention (Zimmerman & Kitsantas, 2017)."
    );
  else
    feedbackMessages.push(
      "Excellent accuracy! You’re building consistent motor control."
    );

  if (report.wpm < 25)
    feedbackMessages.push(
      "Try maintaining a smooth rhythm — consistency matters more than raw speed (Feit et al., 2017)."
    );
  else
    feedbackMessages.push(
      "Great speed! Keep refining your finger technique for comfort and control."
    );

  if (totalMistakes > 5 && mostErrorFinger)
    feedbackMessages.push(
      `Watch your ${mostErrorFinger.replace(
        "_",
        " "
      )} finger — keep it near the home row to reduce strain (ISO 9241-410).`
    );

  if (report.streaks >= 5)
    feedbackMessages.push(
      "Excellent streaks! You’re developing rhythm and automaticity (Hamari et al., 2016)."
    );

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="fixed inset-0 flex items-center justify-center bg-black/50 z-50"
    >
      <PixelCard className="p-6 max-w-lg w-full text-center space-y-3">
        <h2 className="text-2xl font-bold mb-2">Session Summary</h2>
        <div className="space-y-1">
          <p>Accuracy: {report.accuracy.toFixed(1)}%</p>
          <p>WPM: {report.wpm}</p>
          <p>Correct Keystrokes: {report.correct_keystrokes}</p>
          <p>Incorrect Keystrokes: {report.incorrect_keystrokes}</p>
          <p>Duration: {report.session_duration_sec}s</p>
        </div>

        <div className="mt-4 text-left text-sm space-y-1">
          <p className="font-semibold">Feedback:</p>
          <ul className="list-disc pl-5">
            {feedbackMessages.map((msg, i) => (
              <li key={i}>{msg}</li>
            ))}
          </ul>
        </div>

        <PixelButton variant="orange" className="mt-5" onClick={onClose}>
          Continue
        </PixelButton>
      </PixelCard>
    </motion.div>
  );
}
