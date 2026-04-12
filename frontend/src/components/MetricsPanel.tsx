import { PixelCard } from "./PixelCard";

interface MetricsPanelProps {
  correctCount: number;
  incorrectCount: number;
  correctKeysCount: number;
  wpm: number;
  accuracy: number;
  timeLeft: number;
  timerDuration: number;
}

export const MetricsPanel = ({
  correctCount,
  incorrectCount,
  correctKeysCount,
  wpm,
  accuracy,
  timeLeft,
  timerDuration,
}: MetricsPanelProps) => {
  return (
    <aside className="flex flex-col gap-2">
      <PixelCard variant="green" className="p-2.5 text-center">
        <p className="font-pixel text-[9px] uppercase tracking-widest text-white/70 mb-0.5">Correct</p>
        <p className="font-pixel text-2xl text-white drop-shadow">{correctCount}</p>
      </PixelCard>

      <PixelCard variant="red" className="p-2.5 text-center">
        <p className="font-pixel text-[9px] uppercase tracking-widest text-white/70 mb-0.5">Incorrect</p>
        <p className="font-pixel text-2xl text-white drop-shadow">{incorrectCount}</p>
      </PixelCard>

      <PixelCard variant="blue" className="p-2.5 text-center">
        <p className="font-pixel text-[9px] uppercase tracking-widest text-white/70 mb-0.5">Correct Keys</p>
        <p className="font-pixel text-2xl text-white drop-shadow">{correctKeysCount}</p>
      </PixelCard>

      <PixelCard variant="yellow" className="p-2.5 text-center">
        <p className="font-pixel text-[9px] uppercase tracking-widest text-white/70 mb-0.5">Gross WPM</p>
        <p className="font-pixel text-2xl text-white drop-shadow transition-all duration-300">{wpm.toFixed(1)}</p>
      </PixelCard>
  
      <PixelCard variant="orange" className="p-2.5 text-center">
        <p className="font-pixel text-[9px] uppercase tracking-widest text-white/70 mb-0.5">Accuracy</p>
        <p className="font-pixel text-2xl text-white drop-shadow transition-all duration-300">{accuracy.toFixed(1)}%</p>
      </PixelCard>

      <PixelCard
        variant="purple"
        className={`p-3 text-center relative overflow-hidden ${timeLeft <= 10 ? "animate-pulse" : ""}`}
      >
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div
            className="rounded-full transition-all duration-1000"
            style={{
              width: `${(timeLeft / timerDuration) * 64}px`,
              height: `${(timeLeft / timerDuration) * 64}px`,
              background: `radial-gradient(circle, rgba(178,69,146,0.35) 0%, transparent 70%)`,
            }}
          />
        </div>
        <p className="font-pixel text-[9px] uppercase tracking-widest text-white/70 mb-0.5 relative">Timer</p>
        <p
          className={`font-pixel text-3xl text-white drop-shadow relative ${timeLeft <= 10 ? "text-red-200" : ""
            }`}
        >
          {timeLeft < 10 ? "0" : ""}{timeLeft}
        </p>
        <p className="font-pixel text-[8px] text-white/50 mt-0.5 relative">seconds</p>
      </PixelCard>
    </aside>
  );
};
