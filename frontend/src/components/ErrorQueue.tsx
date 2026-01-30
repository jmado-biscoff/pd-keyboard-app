import { PixelCard } from "./PixelCard";

interface ErrorQueueEntry {
  id: number;
  type: "incorrect_key" | "incorrect_finger";
  description: string;
}

interface ErrorQueueProps {
  errorQueue: ErrorQueueEntry[];
}

export const ErrorQueue = ({ errorQueue }: ErrorQueueProps) => {
  return (
    <aside className="flex flex-col gap-1.5">
      <div className="font-pixel text-[8px] uppercase tracking-widest text-muted-foreground/50 mb-1 px-0.5">
        ⚠️ Recent Errors
      </div>
      {[...errorQueue].reverse().map((entry, idx) => (
        <PixelCard
          key={entry.id}
          className={`p-2 transition-all duration-300 ${
            idx === 0
              ? "border-red-500/40 bg-red-900/15 animate-slideDown"
              : "border-border/30"
          }`}
        >
          <div className="flex items-center gap-1 mb-0.5">
            <span
              className={`font-pixel text-[7px] uppercase tracking-wider px-1 py-0.5 rounded ${
                entry.type === "incorrect_key"
                  ? "bg-red-500/20 text-red-300"
                  : "bg-orange-500/20 text-orange-300"
              }`}
            >
              {entry.type === "incorrect_key" ? "Wrong Key" : "Wrong Finger"}
            </span>
          </div>
          <p className="font-pixel text-[8px] text-muted-foreground leading-snug">
            {entry.description}
          </p>
        </PixelCard>
      ))}

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
