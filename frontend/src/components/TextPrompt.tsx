import { useState, useEffect } from "react";

interface TextPromptProps {
  words: string[];
  currentWordIndex: number;
  userInput: string;
  charFeedback: { [index: number]: string };
  level: number;
}

export const TextPrompt = ({
  words,
  currentWordIndex,
  userInput,
  charFeedback,
  level,
}: TextPromptProps) => {

  // ════════════════════════════════════════════════════════════
  // CONFIGURATION: Character spacing and smooth scrolling
  // ════════════════════════════════════════════════════════════
  const CHAR_WIDTH = level === 1 ? 40 : 22;
  const SCROLL_DURATION = 200;

  // SMART JOINING: Level 1 = no spaces, Level 2+ = spaces between words
  const fullText = words.join(level === 1 ? "" : " ");

  const globalCursorPos = (() => {
    let pos = 0;
    for (let i = 0; i < currentWordIndex; i++) {
      pos += (words[i]?.length || 0) + (level === 1 ? 0 : 1);
    }
    pos += userInput.length;
    return pos;
  })();

  const charsBefore = fullText.slice(0, globalCursorPos);
  const charAtCursor = fullText[globalCursorPos] || "";
  const charsAfter = fullText.slice(globalCursorPos + 1);

  const [shakeIndices, setShakeIndices] = useState<Set<number>>(new Set());

  useEffect(() => {
    const incorrectIndices = Object.entries(charFeedback)
      .filter(([, v]) => v === "incorrect" || v === "wrong_finger")
      .map(([k]) => parseInt(k))
      .filter((idx) => Math.abs(idx - globalCursorPos) <= 3);

    if (incorrectIndices.length > 0) {
      const newSet = new Set(incorrectIndices);
      setShakeIndices(newSet);
      const timeout = setTimeout(() => setShakeIndices(new Set()), 400);
      return () => clearTimeout(timeout);
    } else {
      setShakeIndices(new Set());
    }
  }, [charFeedback, globalCursorPos]);

  const getColorClass = (feedback: string | undefined) => {
    if (feedback === "correct") return "text-green-400 bg-green-500/15";
    if (feedback === "wrong_finger") return "text-orange-400 bg-orange-500/15";
    if (feedback === "incorrect") return "text-red-400 bg-red-500/15";
    return "text-muted-foreground/30";
  };

  return (
    <div className="w-full max-w-2xl">
      <div
        className="relative overflow-hidden rounded-lg bg-transparent backdrop-blur-sm border border-border/40"
        style={{ height: "6rem" }}
      >
        <div className="absolute top-0 bottom-0 left-1/2 w-px bg-purple-500/50 shadow-[0_0_6px_rgba(178,69,146,0.4)]" />
        <div className="absolute top-1 bottom-1 left-1/2 w-0.5 bg-purple-500 animate-pulse rounded-full" />

        <div
          className="absolute whitespace-nowrap transition-all ease-out"
          style={{
            top: "50%",
            left: "50%",
            transform: `translate3d(-${globalCursorPos * CHAR_WIDTH + CHAR_WIDTH / 2}px, -50%, 0)`,
            transitionDuration: `${SCROLL_DURATION}ms`,
            willChange: "transform",
          }}
        >
          {(() => {
            let strictIndex = 0;
            return charsBefore.split("").map((ch, visualIndex) => {
              const feedback = ch === " " ? undefined : charFeedback[strictIndex];

              if (ch !== " ") {
                strictIndex++;
              }

              const colorClass = getColorClass(feedback);

              return (
                <span
                  key={`b-${visualIndex}`}
                  className={`font-pixel text-2xl ${colorClass} inline-block transition-colors duration-150 rounded-sm`}
                  style={{ width: `${CHAR_WIDTH}px`, textAlign: "center" }}
                >
                  {ch === " " ? "\u00A0" : ch}
                </span>
              );
            });
          })()}

          <span
            className="font-pixel text-2xl text-foreground inline-block bg-purple-500/25 border-b-2 border-purple-500 rounded-sm"
            style={{ width: `${CHAR_WIDTH}px`, textAlign: "center" }}
          >
            {charAtCursor === " " ? "\u00A0" : charAtCursor}
          </span>

          {charsAfter.split("").map((ch, i) => {
            const globalIdx = globalCursorPos + 1 + i;
            const isShaking = shakeIndices.has(globalIdx);
            return (
              <span
                key={`a-${i}`}
                className={`font-pixel text-2xl text-foreground inline-block ${isShaking ? "animate-bounce text-red-500" : ""
                  }`}
                style={{ width: `${CHAR_WIDTH}px`, textAlign: "center" }}
              >
                {ch === " " ? "\u00A0" : ch}
              </span>
            );
          })}
        </div>
      </div>
    </div>
  );
};
