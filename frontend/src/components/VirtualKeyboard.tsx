// ============================================================
// VIRTUAL KEYBOARD — BINARY COLOR FEEDBACK SYSTEM
// ============================================================
// This component renders a visual keyboard with real-time feedback
// using a simplified two-color system:
//
// - GREEN: Key press is FULLY CORRECT (right key + right finger)
// - RED: Key press has ANY ERROR (wrong key OR wrong finger)
//
// The activeKeys prop contains key-color mappings from PlaySession,
// where each key is either "green" (perfect) or "red" (mistake).
// This binary feedback helps students quickly understand: green means
// perfect execution, red means something needs correction.
// ============================================================

interface VirtualKeyboardProps {
  activeKeys: { [key: string]: string };
  isCalibrating?: boolean;
}

const keyboardLayout = [
  ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
  ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
  ["Z", "X", "C", "V", "B", "N", "M"],
];

const fingerKeyMap: Record<string, string[]> = {
  "left-pinky": ["Q", "A", "Z"],
  "left-ring": ["W", "S", "X"],
  "left-middle": ["E", "D", "C"],
  "left-index": ["R", "T", "F", "G", "V", "B"],
  "right-index": ["Y", "U", "H", "J", "N", "M"],
  "right-middle": ["I", "K"],
  "right-ring": ["O", "L"],
  "right-pinky": ["P"],
};

const getFingerForKey = (key: string): string => {
  for (const [finger, keys] of Object.entries(fingerKeyMap)) {
    if (keys.includes(key)) return finger;
  }
  return "";
};

// Unified colors per finger type (background color when idle)
const fingerColorsUnified: Record<string, string> = {
  "pinky": "bg-blue-900",
  "ring": "bg-blue-700",
  "middle": "bg-blue-500",
  "index": "bg-blue-300",
};

const mapFingerToType = (finger: string): string => {
  if (finger.includes("pinky")) return "pinky";
  if (finger.includes("ring")) return "ring";
  if (finger.includes("middle")) return "middle";
  if (finger.includes("index")) return "index";
  return "unknown";
};

export const VirtualKeyboard = ({ activeKeys, isCalibrating }: VirtualKeyboardProps) => {
  // During calibration, ignore all active keys to prevent false green highlights
  const effectiveKeys = isCalibrating ? {} : activeKeys;
  return (
    <div className="flex flex-col items-center">
      <div className="font-pixel text-[8px] uppercase tracking-widest text-muted-foreground/60 mb-1.5">
      </div>
      <div className="flex flex-col items-center gap-2 bg-card/30 rounded-lg px-4 py-3 border border-border/30 shadow-inner">
        {keyboardLayout.map((row, rowIdx) => (
          <div key={rowIdx} className="flex gap-2 justify-center">
            {row.map((key) => {
              const keyState = effectiveKeys[key] || "";
              const isActive = !!effectiveKeys[key];

              // ============================================================
              // BINARY COLOR LOGIC: GREEN for perfect, RED for any mistake
              // ============================================================
              // - GREEN: Correct key pressed with correct finger
              // - RED: Wrong key OR correct key with wrong finger
              //
              // No intermediate states (orange, yellow, etc.) — this keeps
              // feedback simple and actionable for the student.
              // ============================================================
              let highlightClass = "";
              if (isActive) {
                if (keyState.includes("green")) {
                  // Fully correct key press
                  highlightClass =
                    "bg-green-500 text-white border-green-600 shadow-[0_0_10px_rgba(34,197,94,0.6)]";
                } else {
                  // Any kind of mistake (wrong key OR wrong finger)
                  highlightClass =
                    "bg-red-500 text-white border-red-600 shadow-[0_0_10px_rgba(239,68,68,0.6)]";
                }
              }

              const finger = getFingerForKey(key);
              const fingerType = mapFingerToType(finger);
              const baseColor = fingerColorsUnified[fingerType] || "bg-muted";

              return (
                <div
                  key={key}
                  className={`
                    pixel-border w-11 h-11 flex items-center justify-center font-pixel text-sm rounded-md
                    shadow-[0_2px_4px_rgba(0,0,0,0.2)] border
                    transition-all duration-150 select-none
                    ${isActive ? highlightClass : `${baseColor} text-white border-blue-600/50`}
                  `}
                >
                  {key}
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
};
