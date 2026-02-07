import { useState, useEffect } from "react";

interface TextPromptProps {
  words: string[];
  currentWordIndex: number;
  userInput: string;
  charFeedback: { [index: number]: string };
  level: number; // ✅ Added to support smart spacing
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
  // ✅ LEVEL 1: Extra spacing for visual separation (no logical spaces)
  const CHAR_WIDTH = level === 1 ? 36 : 18; // Level 1: 36px for visual spacing
  const SCROLL_DURATION = 200; // Smooth scroll duration in ms

  // ✅ SMART JOINING: Level 1 = no spaces (letters only)
  //                   Level 2+ = spaces between words
  const fullText = words.join(level === 1 ? "" : " ");

  const globalCursorPos = (() => {
    let pos = 0;
    for (let i = 0; i < currentWordIndex; i++) {
      // ✅ Only add space separator for levels 2-4
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
    // Find incorrect characters near the current position for shake animation
    const incorrectIndices = Object.entries(charFeedback)
      .filter(([, v]) => v === "incorrect")
      .map(([k]) => parseInt(k))
      .filter((idx) => Math.abs(idx - globalCursorPos) <= 3); // Only shake nearby chars

    if (incorrectIndices.length > 0) {
      const newSet = new Set(incorrectIndices);
      setShakeIndices(newSet);
      const timeout = setTimeout(() => setShakeIndices(new Set()), 400);
      return () => clearTimeout(timeout);
    } else {
      setShakeIndices(new Set());
    }
  }, [charFeedback, globalCursorPos]);

  return (
    <div className="w-full max-w-2xl">
      <div
        className="relative overflow-hidden rounded-lg bg-transparent backdrop-blur-sm border border-border/40"
        style={{ height: "5rem" }}
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
            // ✅ Strict Index Counter - only counts non-space characters to match AI pointer
            let strictIndex = 0;
            return charsBefore.split("").map((ch, visualIndex) => {
              // Only look up feedback for non-space characters using strict index
              const feedback = ch === " " ? undefined : charFeedback[strictIndex];

              // Increment strict counter only for non-space characters
              if (ch !== " ") {
                strictIndex++;
              }

              const colorClass = feedback === "correct"
                ? "text-green-400" // #4ade80
                : feedback === "incorrect"
                  ? "text-red-400" // #f87171
                  : "text-muted-foreground/30";

              return (
                <span
                  key={`b-${visualIndex}`}
                  className={`font-pixel text-xl ${colorClass} inline-block transition-colors duration-150`}
                  style={{ width: `${CHAR_WIDTH}px`, textAlign: "center" }}
                >
                  {ch === " " ? "\u00A0" : ch}
                </span>
              );
            });
          })()}

          <span
            className="font-pixel text-xl text-foreground inline-block bg-purple-500/25 border-b-2 border-purple-500 rounded-sm"
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
                className={`font-pixel text-xl text-foreground inline-block ${isShaking ? "animate-bounce text-red-500" : ""
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
