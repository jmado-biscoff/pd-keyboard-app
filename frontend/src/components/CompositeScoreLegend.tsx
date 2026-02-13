interface CompositeScoreLegendProps {
  currentGrade: string;
}

const ratings = [
  { grade: "A", label: "Excellent", color: "bg-green-600" },
  { grade: "B", label: "Competent", color: "bg-green-400" },
  { grade: "C", label: "Developing", color: "bg-orange-500" },
  { grade: "D", label: "Beginner", color: "bg-red-400" },
  { grade: "E", label: "Error-Dominant", color: "bg-red-700" },
];

export const CompositeScoreLegend = ({ currentGrade }: CompositeScoreLegendProps) => {
  return (
    <aside className="flex flex-col gap-1.5">
      <div className="font-pixel text-[8px] uppercase tracking-widest text-muted-foreground/50 mb-1 px-0.5">
        Composite Score Legend
      </div>
      <div className="flex flex-col gap-2">
        {ratings.map((r) => {
          const isActive = r.grade === currentGrade;
          return (
            <div
              key={r.grade}
              className={`flex items-center gap-3 rounded-md px-2 py-1.5 transition-all ${
                isActive
                  ? "ring-2 ring-accent scale-105 bg-card/40"
                  : "opacity-60"
              }`}
            >
              <div
                className={`w-9 h-9 ${r.color} rounded-md flex items-center justify-center flex-shrink-0 border border-black/20`}
              >
                <span className="font-pixel text-lg text-black font-bold">
                  {r.grade}
                </span>
              </div>
              <span className="font-pixel text-xs text-foreground">
                {r.label}
              </span>
            </div>
          );
        })}
      </div>
    </aside>
  );
};
