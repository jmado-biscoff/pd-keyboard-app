import { useState, useEffect, useMemo } from "react";
import { PixelCard } from "./PixelCard";
import { PixelButton } from "./PixelButton";
import { analyzeSession } from "@/utils/displayBrain";
import type { ErrorHistoryEntry, SessionHistoryEntry, SessionAnalysis } from "@/types/typing";

import { KEYBOARD_IMAGES } from "@/utils/keyboardImages";

// ── Static imports: celebration GIFs ──
import gifDog from "@/assets/gifs/dog.gif";
import gifBunny from "@/assets/gifs/bunny.gif";
import gifDeer from "@/assets/gifs/deer.gif";
import gifPenguin from "@/assets/gifs/penguin.gif";
import gifChick from "@/assets/gifs/chick.gif";
import gifCat from "@/assets/gifs/cat.gif";

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
      A: "Meow! Purrfect! Your paws hit every key with the right finger—flawless technique! You've mastered the art of touch typing!",
      B: "Meow! Good work! Your finger placement is solid, but a few paws slipped. Keep practicing those tricky keys!",
      C: "Meow... Getting there! You're learning which finger goes where, but too many wrong paws touched the keys. Focus on finger discipline!",
      D: "Meow. Beginner paws! You're still figuring out which finger belongs on which key. Slow down and memorize the finger-to-key map!",
      E: "Hiss! Too many wrong fingers! You pressed keys with the wrong paws way too often. Review proper finger placement before continuing!",
    },
  },
  {
    name: "Dog",
    gif: gifDog,
    greetings: {
      A: "Woof! Top dog! Every key pressed with the perfect finger—your paw-to-key technique is outstanding!",
      B: "Woof! Good boy/girl! Most keys hit with the right finger, but a few paws wandered. Tighten up that finger discipline!",
      C: "Ruff! Learning the ropes! You're getting better at finger placement, but still using wrong fingers too often. Keep training!",
      D: "Woof... Puppy paws! You're still learning which finger should press which key. Practice the home row finger positions first!",
      E: "Ruff ruff! Bad paw placement! Way too many keys pressed with incorrect fingers. Review the finger-to-key chart and try again!",
    },
  },
  {
    name: "Bunny",
    gif: gifBunny,
    greetings: {
      A: "Hop hop! Lightning paws! Every key tapped with the correct finger—your technique is hopping perfect!",
      B: "Hop! Quick learner! Most keys hit with proper fingers, but a few hops went astray. Refine those finger assignments!",
      C: "Thump thump! Bouncing along! You're improving finger awareness, but still hopping to wrong keys with wrong fingers. Focus!",
      D: "Hop... Baby bunny! You're just starting to learn which finger presses which key. Take it slow and build muscle memory!",
      E: "Oh no! Chaotic hops! Your fingers are all over the place—too many wrong finger presses. Study proper finger placement!",
    },
  },
  {
    name: "Chick",
    gif: gifChick,
    greetings: {
      A: "Peep peep! Precise pecks! Every key pecked with the exact right finger—your technique is top-notch!",
      B: "Peep! Growing strong! Most pecks used the correct finger, but a few went to the wrong keys. Keep practicing!",
      C: "Chirp! Hatching skills! You're learning finger assignments, but too many pecks used wrong fingers. Focus on accuracy!",
      D: "Chirp... Little chick! You're still figuring out which finger pecks which key. Start with home row and build from there!",
      E: "Chirp chirp! Scattered pecks! Way too many keys pecked with incorrect fingers. Review finger placement basics!",
    },
  },
  {
    name: "Deer",
    gif: gifDeer,
    greetings: {
      A: "Oh deer! Graceful gallop! Every key pressed with perfect finger form—your technique is majestic!",
      B: "Oh deer! Steady stride! Most keys hit with proper fingers, but a few hooves slipped. Tighten your finger control!",
      C: "Oh deer! Trotting along! You're learning finger positions, but stumbling with wrong fingers too often. Keep practicing!",
      D: "Oh deer... Fawn steps! You're just learning which finger belongs on which key. Take it one step at a time!",
      E: "Oh deer! Lost in the forest! Too many keys pressed with wrong fingers—you need to review proper finger assignments!",
    },
  },
  {
    name: "Penguin",
    gif: gifPenguin,
    greetings: {
      A: "Waddle you know! Ice-cold precision! Every key tapped with the perfect finger—flawless flipper technique!",
      B: "Waddle! Smooth glide! Most keys hit with correct fingers, but a few flippers slipped. Refine your technique!",
      C: "Waddle waddle! Finding footing! You're learning finger placement, but too many wrong flipper presses. Focus on form!",
      D: "Waddle... Baby penguin! You're still learning which flipper presses which key. Practice home row positions first!",
      E: "Brrr! Slippery ice! Way too many keys pressed with wrong fingers—you need to review proper finger-to-key mapping!",
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
  /** Physical key accuracy count: correct letters pressed regardless of finger technique */
  correctKeysCount?: number;
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
  correctKeysCount,
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

  const analysis = finalAnalysis?.analysis || analyzeSession(wpm, accuracy, correctCount, incorrectCount, errorHistory, correctKeysCount ?? correctCount);
  const dbMetrics = finalAnalysis?.dbMetrics || {
    wpm,
    netWpm: analysis.netWpm,
    accuracy: analysis.accuracyPercent,
    errorRate: analysis.errorRate,
    compositeScore: analysis.compositeScore,
    completionRate: analysis.completionRate,
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
          POSITION 1: 5-Column Metrics Panel
          Accuracy = physical key accuracy (correct letters / total)
          Error Rate = technique errors (wrong key OR wrong finger / total)
          These are intentionally decoupled: hitting the right letter with
          the wrong finger raises Error Rate but keeps Accuracy at 100%.
          ═══════════════════════════════════════════════════════════════ */}
      <div className="grid grid-cols-5 gap-2 max-w-xl mx-auto mb-6">
        <div className="rounded-lg bg-purple-500/15 border border-purple-500/30 flex flex-col items-center justify-center py-4 min-h-[80px]">
          <p className="font-pixel text-[7px] text-purple-400/80 uppercase tracking-wider mb-1">Rating</p>
          <p className="font-pixel text-xl text-purple-400">
            {isCalculating ? "-" : analysis.letterGrade}
          </p>
        </div>

        <div className="rounded-lg bg-green-500/15 border border-green-500/30 flex flex-col items-center justify-center py-4 min-h-[80px]">
          <p className="font-pixel text-[7px] text-green-400/80 uppercase tracking-wider mb-1">Score</p>
          <p className="font-pixel text-xl text-green-400">
            {isCalculating ? "-" : dbMetrics.compositeScore.toFixed(1)}
          </p>
        </div>

        <div className="rounded-lg bg-blue-500/15 border border-blue-500/30 flex flex-col items-center justify-center py-4 min-h-[80px]">
          <p className="font-pixel text-[7px] text-blue-400/80 uppercase tracking-wider mb-1">Net WPM</p>
          <p className="font-pixel text-xl text-blue-400">
            {isCalculating ? "-" : dbMetrics.netWpm}
          </p>
        </div>

        <div className="rounded-lg bg-yellow-500/15 border border-yellow-500/30 flex flex-col items-center justify-center py-4 min-h-[80px]">
          <p className="font-pixel text-[7px] text-yellow-400/80 uppercase tracking-wider mb-1">Accuracy</p>
          <p className="font-pixel text-xl text-yellow-400">
            {isCalculating ? "-" : `${analysis.accuracyPercent.toFixed(1)}%`}
          </p>
        </div>

        <div className="rounded-lg bg-red-500/15 border border-red-500/30 flex flex-col items-center justify-center py-4 min-h-[80px]">
          <p className="font-pixel text-[7px] text-red-400/80 uppercase tracking-wider mb-1">Err Rate</p>
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
