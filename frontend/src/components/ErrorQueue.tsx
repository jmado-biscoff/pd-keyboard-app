import { PixelCard } from "./PixelCard";
import type { ErrorQueueEntry } from "@/types/typing";

interface ErrorQueueProps {
  errorQueue: ErrorQueueEntry[];
}

export const ErrorQueue = ({ errorQueue }: ErrorQueueProps) => {
  return (
    <aside className="flex flex-col gap-1.5">
      <div className="font-pixel text-[8px] uppercase tracking-widest text-muted-foreground/50 mb-1 px-0.5">
        Recent Errors
      </div>
      {[...errorQueue].reverse().map((entry, idx) => {
        const isWrongKey = entry.type === "incorrect_key";
        const bgColor = isWrongKey ? "bg-red-500" : "bg-orange-500";

        return (
          <div
            key={entry.id}
            className={`flex items-stretch gap-2 transition-all duration-300 ${idx === 0 ? "animate-slideDown" : ""
              }`}
          >
            {/* Left: Pressed Key Box with Solid Colored Background */}
            <div className={`flex-shrink-0 w-10 h-10 border-2 border-white/30 ${bgColor} rounded flex items-center justify-center shadow-md`}>
              <span className="font-pixel text-lg text-white uppercase">
                {entry.pressedKey}
              </span>
            </div>

            {/* Right: Error Message with Solid Color Background */}
            <div className={`flex-1 ${bgColor} rounded px-3 py-2 flex flex-col justify-center`}>
              <div className="font-pixel text-[8px] text-white uppercase tracking-wider mb-0.5">
                {isWrongKey ? "🔴 Wrong Key" : "🟠 Wrong Finger"}
              </div>
              <p className="font-pixel text-[9px] text-white leading-tight">
                {entry.description}
              </p>
            </div>
          </div>
        );
      })}

      {errorQueue.length === 0 && (
        <PixelCard className="p-2.5 opacity-25">
          <p className="font-pixel text-[8px] text-muted-foreground text-center">
            No errors yet
          </p>
        </PixelCard>
      )}
    </aside>
  );
};
