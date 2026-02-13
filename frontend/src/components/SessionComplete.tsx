import { useState, useEffect, useMemo } from "react";
import { PixelCard } from "./PixelCard";
import { PixelButton } from "./PixelButton";
import { analyzeSession } from "@/utils/displayBrain";
import type { ErrorHistoryEntry, SessionHistoryEntry, SessionAnalysis } from "@/types/typing";

// ── Static imports: proper-keyboard images (A-Z) ──
import imgA from "@/assets/proper-keyboard/A.png";
import imgB from "@/assets/proper-keyboard/B.png";
import imgC from "@/assets/proper-keyboard/C.png";
import imgD from "@/assets/proper-keyboard/D.png";
import imgE from "@/assets/proper-keyboard/E.png";
import imgF from "@/assets/proper-keyboard/F.png";
import imgG from "@/assets/proper-keyboard/G.png";
import imgH from "@/assets/proper-keyboard/H.png";
import imgI from "@/assets/proper-keyboard/I.png";
import imgJ from "@/assets/proper-keyboard/J.png";
import imgK from "@/assets/proper-keyboard/K.png";
import imgL from "@/assets/proper-keyboard/L.png";
import imgM from "@/assets/proper-keyboard/M.png";
import imgN from "@/assets/proper-keyboard/N.png";
import imgO from "@/assets/proper-keyboard/O.png";
import imgP from "@/assets/proper-keyboard/P.png";
import imgQ from "@/assets/proper-keyboard/Q.png";
import imgR from "@/assets/proper-keyboard/R.png";
import imgS from "@/assets/proper-keyboard/S.png";
import imgT from "@/assets/proper-keyboard/T.png";
import imgU from "@/assets/proper-keyboard/U.png";
import imgV from "@/assets/proper-keyboard/V.png";
import imgW from "@/assets/proper-keyboard/W.png";
import imgX from "@/assets/proper-keyboard/X.png";
import imgY from "@/assets/proper-keyboard/Y.png";
import imgZ from "@/assets/proper-keyboard/Z.png";

// ── Static imports: celebration GIFs ──
import gifDog from "@/assets/gifs/dog.gif";
import gifBunny from "@/assets/gifs/bunny.gif";
import gifDeer from "@/assets/gifs/deer.gif";
import gifPenguin from "@/assets/gifs/penguin.gif";
import gifChick from "@/assets/gifs/chick.gif";
import gifCat from "@/assets/gifs/cat.gif";

const KEYBOARD_IMAGES: Record<string, string> = {
  A: imgA, B: imgB, C: imgC, D: imgD, E: imgE, F: imgF,
  G: imgG, H: imgH, I: imgI, J: imgJ, K: imgK, L: imgL,
  M: imgM, N: imgN, O: imgO, P: imgP, Q: imgQ, R: imgR,
  S: imgS, T: imgT, U: imgU, V: imgV, W: imgW, X: imgX,
  Y: imgY, Z: imgZ,
};

// ── Animal character personalities ──
interface AnimalCharacter {
  name: string;
  gif: string;
  greetings: Record<string, string>;
}

const ANIMAL_CHARACTERS: AnimalCharacter[] = [
  {
    name: "Cat",
    gif: gifCat,
    greetings: {
      A: "Meow! Excellent! Outstanding speed, accuracy, and finger technique. You've earned a catnap!",
      B: "Meow! Competent! Strong paws on those keys. A little more practice and you'll be purr-fect!",
      C: "Meow... Developing. Keep sharpening those claws on the keyboard. Fewer errors will get you there!",
      D: "Meow. Beginner. Even kittens start small. Focus on placing your paws on the right keys first!",
      E: "Hiss! Error-Dominant. Too many missed keys! Slow down and focus on using the correct fingers.",
    },
  },
  {
    name: "Dog",
    gif: gifDog,
    greetings: {
      A: "Woof! Excellent! You're top dog! Outstanding speed, accuracy, and finger technique!",
      B: "Woof! Competent! Solid balance across the board. Keep fetching those high scores!",
      C: "Ruff! Developing. You're learning new tricks! Focus on reducing errors and building consistency.",
      D: "Woof... Beginner. Every pup starts somewhere! Take it slow and learn the right finger positions.",
      E: "Ruff ruff! Error-Dominant. Too many wrong keys! Slow down and sniff out the correct fingers.",
    },
  },
  {
    name: "Bunny",
    gif: gifBunny,
    greetings: {
      A: "Hop hop! Excellent! Lightning-fast paws with amazing accuracy! You're hopping to the top!",
      B: "Hop! Competent! Great balance of speed and accuracy. Keep bouncing toward mastery!",
      C: "Thump thump! Developing. You're making progress! Fewer hops in the wrong direction will help.",
      D: "Hop... Beginner. Every bunny starts somewhere! Take your time and find the right keys first.",
      E: "Oh no! Error-Dominant. Too many wrong hops! Slow down and focus on placing your paws correctly.",
    },
  },
  {
    name: "Chick",
    gif: gifChick,
    greetings: {
      A: "Peep peep! Excellent! Your pecking speed and accuracy are top-notch! Great work!",
      B: "Peep! Competent! You're hatching into a great typist. Keep pecking at those keys!",
      C: "Chirp! Developing. You're cracking out of your shell! Focus on accurate pecks to grow faster.",
      D: "Chirp... Beginner. Little chicks need time to grow. Focus on finding the right keys with each peck!",
      E: "Chirp chirp! Error-Dominant. Too many missed pecks! Slow down and aim for the right keys.",
    },
  },
  {
    name: "Deer",
    gif: gifDeer,
    greetings: {
      A: "Oh deer! Excellent! You galloped through that with amazing speed and grace! Truly majestic!",
      B: "Oh deer! Competent! Strong strides across the keyboard. Keep galloping toward the finish line!",
      C: "Oh deer! Developing. You're trotting along nicely! Fewer stumbles will help you pick up speed.",
      D: "Oh deer... Beginner. Take it one step at a time. Build your footing on the right keys first!",
      E: "Oh deer! Error-Dominant. Too many stumbles on the trail! Slow your pace and find the right path.",
    },
  },
  {
    name: "Penguin",
    gif: gifPenguin,
    greetings: {
      A: "Waddle you know! Excellent! Ice-cold precision and speed! You're sliding to victory!",
      B: "Waddle! Competent! Smooth gliding across the keys. Keep sliding toward the top!",
      C: "Waddle waddle! Developing. You're finding your footing on the ice. Fewer slips and you'll soar!",
      D: "Waddle... Beginner. The ice is slippery at first! Take it slow and learn the right flipper positions.",
      E: "Brrr! Error-Dominant. Too many slips on the ice! Slow down and focus on steady flipper placement.",
    },
  },
];

/** Build dynamic speech bubble body text based on the current keystroke */
function getKeystrokeFeedback(entry: SessionHistoryEntry | undefined): string {
  if (!entry) return "";
  if (entry.status === "correct") {
    return `You pressed "${entry.char}" with the correct finger. Nice!`;
  }
  if (entry.status === "skipped") {
    return `You didn't reach "${entry.expected}" in time. Try to keep up the pace next round!`;
  }
  if (entry.status === "wrong_finger") {
    if (entry.hand && entry.finger) {
      return `You pressed "${entry.char}" with your ${entry.hand} ${entry.finger}, but should use the correct finger for "${entry.expected}".`;
    }
    return `You pressed "${entry.char}" but used the wrong finger for "${entry.expected}".`;
  }
  // wrong_key
  return `You pressed "${entry.char}" instead of "${entry.expected}". Watch your finger placement!`;
}

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
  replayIndex: number;
  onReplayIndexChange: (index: number) => void;
}

export const SessionComplete = ({
  wpm,
  accuracy,
  correctCount,
  incorrectCount,
  errorHistory,
  sessionHistory,
  onFinish,
  finalAnalysis,
  replayIndex,
  onReplayIndexChange,
}: SessionCompleteProps) => {
  const [isCalculating, setIsCalculating] = useState(true);

  const character = useMemo(
    () => ANIMAL_CHARACTERS[Math.floor(Math.random() * ANIMAL_CHARACTERS.length)],
    []
  );

  useEffect(() => {
    const timer = setTimeout(() => setIsCalculating(false), 500);
    return () => clearTimeout(timer);
  }, []);

  const analysis = finalAnalysis?.analysis || analyzeSession(wpm, accuracy, correctCount, incorrectCount, errorHistory);
  const dbMetrics = finalAnalysis?.dbMetrics || {
    wpm,
    netWpm: analysis.netWpm,
    accuracy: analysis.accuracyPercent,
    errorRate: analysis.errorRate,
    compositeScore: analysis.compositeScore,
  };

  // ════════════════════════════════════════════════════════════
  // TIMELINE SCROLLER LOGIC
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
      isActive: adjustedStart + idx === replayIndex,
    }));
  };

  const visibleEntries = getVisibleEntries();
  const currentEntry = sessionHistory[replayIndex];

  const isCurrentIncorrect = currentEntry && currentEntry.status !== "correct";
  const expectedLetter = currentEntry?.expected?.toUpperCase();
  const properKeyboardImg = expectedLetter ? KEYBOARD_IMAGES[expectedLetter] : undefined;

  // Dynamic speech bubble text for current keystroke
  const keystrokeFeedback = getKeystrokeFeedback(currentEntry);

  return (
    <PixelCard className="text-center py-6 shadow-xl w-full max-w-3xl mx-auto min-h-[700px]">

      {/* ═══════════════════════════════════════════════════════════════
          POSITION 1: 4-Column Metrics Panel
          ═══════════════════════════════════════════════════════════════ */}
      <div className="grid grid-cols-4 gap-3 max-w-md mx-auto mb-6">
        <div className="rounded-lg bg-purple-500/15 border border-purple-500/30 flex flex-col items-center justify-center py-4 min-h-[80px]">
          <p className="font-pixel text-[8px] text-purple-400/80 uppercase tracking-wider mb-1">Rating</p>
          <p className="font-pixel text-xl text-purple-400">
            {isCalculating ? "-" : analysis.letterGrade}
          </p>
        </div>

        <div className="rounded-lg bg-green-500/15 border border-green-500/30 flex flex-col items-center justify-center py-4 min-h-[80px]">
          <p className="font-pixel text-[8px] text-green-400/80 uppercase tracking-wider mb-1">Score</p>
          <p className="font-pixel text-xl text-green-400">
            {isCalculating ? "-" : dbMetrics.compositeScore.toFixed(1)}
          </p>
        </div>

        <div className="rounded-lg bg-blue-500/15 border border-blue-500/30 flex flex-col items-center justify-center py-4 min-h-[80px]">
          <p className="font-pixel text-[8px] text-blue-400/80 uppercase tracking-wider mb-1">Net WPM</p>
          <p className="font-pixel text-xl text-blue-400">
            {isCalculating ? "-" : dbMetrics.netWpm}
          </p>
        </div>

        <div className="rounded-lg bg-red-500/15 border border-red-500/30 flex flex-col items-center justify-center py-4 min-h-[80px]">
          <p className="font-pixel text-[8px] text-red-400/80 uppercase tracking-wider mb-1">Error Rate</p>
          <p className="font-pixel text-xl text-red-400">
            {isCalculating ? "-" : `${dbMetrics.errorRate.toFixed(1)}%`}
          </p>
        </div>
      </div>

      {/* ═══════════════════════════════════════════════════════════════
          SESSION REPLAY SECTION
          ═══════════════════════════════════════════════════════════════ */}
      {sessionHistory.length > 0 && currentEntry && (
        <div className="mb-6 max-w-2xl mx-auto">
          {/* ── TOP: Large Visual Feedback (Image or Green Box) ── */}
          <div className="rounded-lg overflow-hidden mb-4 h-64">
            {isCurrentIncorrect && properKeyboardImg ? (
              <div className="flex items-center justify-center bg-card/30 border border-border/50 rounded-lg p-4 h-full">
                <img
                  src={properKeyboardImg}
                  alt={`Proper finger placement for ${expectedLetter}`}
                  className="max-w-full max-h-full object-scale-down"
                />
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center bg-green-500/15 border border-green-500/30 rounded-lg h-full">
                <p className="font-pixel text-lg text-green-400 mb-1">Correct finger used!</p>
                <p className="font-pixel text-[10px] text-green-400/70">
                  Great job! You used the correct touch typing technique.
                </p>
              </div>
            )}
          </div>

          {/* ── SLIDER: Timeline Scroller ── */}
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
                    className={`transition-all duration-300 ${entry.isActive ? "scale-125 mx-2" : "scale-100 opacity-60"}`}
                  >
                    <div
                      className={`font-pixel text-sm px-3 py-2 rounded ${statusBgColor} ${entry.isActive ? "border-2 border-white shadow-lg" : "border border-white/30"} text-white transition-all duration-200`}
                    >
                      {entry.expected === " " ? "\u2423" : entry.expected}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="mb-6">
            <input
              type="range"
              min="0"
              max={sessionHistory.length - 1}
              value={replayIndex}
              onChange={(e) => onReplayIndexChange(parseInt(e.target.value))}
              className="w-full h-3 bg-card border-2 border-border/50 rounded-lg appearance-none cursor-pointer pixel-slider"
              style={{
                background: `linear-gradient(to right, rgba(34, 197, 94, 0.3) 0%, rgba(34, 197, 94, 0.3) ${(replayIndex / (sessionHistory.length - 1)) * 100}%, rgba(100, 116, 139, 0.2) ${(replayIndex / (sessionHistory.length - 1)) * 100}%, rgba(100, 116, 139, 0.2) 100%)`,
              }}
            />
            <div className="flex justify-between mt-1">
              <span className="font-pixel text-[8px] text-muted-foreground">Keystroke: {replayIndex + 1}</span>
              <span className="font-pixel text-[8px] text-muted-foreground">Total: {sessionHistory.length}</span>
            </div>
          </div>

          {/* ── BOTTOM: Animal Character Speech Bubble ── */}
          <div className="flex items-start gap-4">
            <img
              src={character.gif}
              alt={character.name}
              className="w-28 h-28 object-contain flex-shrink-0 rounded-md"
            />

            <div className="flex-1 relative bg-card/50 border border-border/50 rounded-xl px-4 py-3 text-justify shadow-sm h-36 overflow-y-auto dialogue-scroll">
              {/* Speech bubble arrow */}
              <div
                className="absolute left-[-8px] top-6 w-0 h-0 pointer-events-none"
                style={{
                  borderTop: "8px solid transparent",
                  borderBottom: "8px solid transparent",
                  borderRight: "8px solid hsl(var(--border) / 0.5)",
                }}
              />
              <div
                className="absolute left-[-6px] top-[25px] w-0 h-0 pointer-events-none"
                style={{
                  borderTop: "7px solid transparent",
                  borderBottom: "7px solid transparent",
                  borderRight: "7px solid hsl(var(--card) / 0.5)",
                }}
              />

              {/* Header: animal personality + grade category */}
              <p className="font-pixel text-xs text-foreground mb-1.5 leading-relaxed">
                {character.greetings[analysis.letterGrade] || character.greetings.D}
              </p>

              {/* Dynamic body: keystroke-specific feedback */}
              <p className="font-pixel text-[10px] text-muted-foreground leading-relaxed">
                {keystrokeFeedback}
                {currentEntry.tip ? ` ${currentEntry.tip}` : ""}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Action Button */}
      <PixelButton variant="primary" size="md" className="mt-2" onClick={onFinish}>
        Finish Session
      </PixelButton>

      {/* Slider & Dialogue Styles */}
      <style>{`
        .dialogue-scroll {
          scrollbar-width: none;
          -ms-overflow-style: none;
        }
        .dialogue-scroll::-webkit-scrollbar {
          display: none;
        }
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
