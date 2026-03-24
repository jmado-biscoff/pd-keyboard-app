import { useState, useEffect } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { PixelCard } from "@/components/PixelCard";
import { Loader2 } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
} from "recharts";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:5000/api";

interface StudentAnalyticsModalProps {
  open: boolean;
  onClose: () => void;
  studentId: string;
  studentName: string;
  classroomId: string;
}

interface ResultEntry {
  _id: string;
  level: number;
  wpm: number;
  accuracy: number;
  grade: string;
  sessionType: "practice" | "evaluated";
  compositeScore: number;
  netWpm: number;
  errorRate: number;
  createdAt: string;
  sessionId?: string;
}

interface BarDataPoint {
  label: string;
  compositeScore: number;
  grade: string;
  netWpm: number;
  accuracy: number;
  errorRate: number;
  date: string;
  sessionIndex: number;
  trialIndex: number;
  sessionType: "practice" | "evaluated";
}

function ordinal(n: number): string {
  const s = ["th", "st", "nd", "rd"];
  const v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}

const RAINBOW_COLORS = [
  "#ef4444", // Red
  "#f97316", // Orange
  "#f59e0b", // Amber
  "#eab308", // Yellow
  "#84cc16", // Lime
  "#10b981", // Emerald
  "#06b6d4", // Cyan
  "#3b82f6", // Blue
  "#6366f1", // Indigo
  "#a855f7", // Purple
];

const CustomBarTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null;
  const data = payload[0].payload as BarDataPoint;
  return (
    <div className="bg-black/90 border border-white/20 p-3 rounded font-pixel text-xs text-white">
      <p className="text-yellow-400 mb-1">
        {ordinal(data.sessionIndex)} Session — Trial {data.trialIndex}
      </p>
      <p>Grade: {data.grade || "N/A"}</p>
      <p>Score: {data.compositeScore?.toFixed(1)}</p>
      <p>Net WPM: {Math.round(data.netWpm)}</p>
      <p>Accuracy: {data.accuracy?.toFixed(1)}%</p>
      <p>Error Rate: {data.errorRate?.toFixed(1)}%</p>
      <p>Date: {data.date}</p>
    </div>
  );
};

/**
 * Groups results by sessionId (teacher activation) and returns all trials
 * as flat bar data with session/trial indices.
 */
function buildBarData(results: ResultEntry[], targetLevel: number): BarDataPoint[] {
  const filtered = results
    .filter((r) => r.level === targetLevel && r.sessionType === "evaluated")
    .sort((a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime());

  if (filtered.length === 0) return [];

  // Group by sessionId
  const sessionGroups = new Map<string, ResultEntry[]>();
  filtered.forEach((r) => {
    const sid = r.sessionId || "unknown";
    if (!sessionGroups.has(sid)) sessionGroups.set(sid, []);
    sessionGroups.get(sid)!.push(r);
  });

  // Flatten into bar data points with session + trial labels
  const barData: BarDataPoint[] = [];
  let sIdx = 0;
  for (const [, group] of sessionGroups) {
    group.forEach((r, tIdx) => {
      barData.push({
        label: `S${sIdx + 1}-T${tIdx + 1}`,
        compositeScore: r.compositeScore || 0,
        grade: r.grade || "N/A",
        netWpm: r.netWpm || 0,
        accuracy: r.accuracy || 0,
        errorRate: r.errorRate || 0,
        date: new Date(r.createdAt).toLocaleDateString(),
        sessionIndex: sIdx + 1,
        trialIndex: tIdx + 1,
        sessionType: r.sessionType,
      });
    });
    sIdx++;
  }

  return barData;
}

/** Count unique sessions in the bar data */
function countSessions(data: BarDataPoint[]): number {
  const set = new Set(data.map((d) => d.sessionIndex));
  return set.size;
}

function ProgressChart({ data, title }: { data: BarDataPoint[]; title: string }) {
  if (data.length === 0) {
    return (
      <div className="mb-6">
        <h4 className="font-pixel text-sm mb-2 text-white">{title}</h4>
        <div className="bg-white/5 rounded-lg p-6 text-center">
          <p className="font-pixel text-xs opacity-50 text-white">No sessions recorded yet</p>
        </div>
      </div>
    );
  }

  const sessions = countSessions(data);
  const chartWidth = Math.max(400, sessions * 150);

  return (
    <div className="mb-6">
      <h4 className="font-pixel text-sm mb-2 text-white">{title}</h4>
      <div className="bg-white/5 rounded-lg p-3">
        <div className="h-2" />
        <div className="overflow-x-auto">
          <BarChart
            width={chartWidth}
            height={220}
            data={data}
            margin={{ top: 5, right: 20, left: 0, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis
              dataKey="label"
              stroke="rgba(255,255,255,0.5)"
              tick={{ fontSize: 9, fontFamily: "monospace", fill: "rgba(255,255,255,0.7)" }}
              label={{ value: "Session — Trial", position: "insideBottom", offset: -2, fontSize: 10, fill: "rgba(255,255,255,0.5)" }}
            />
            <YAxis
              domain={[0, 100]}
              stroke="rgba(255,255,255,0.5)"
              tick={{ fontSize: 10, fontFamily: "monospace", fill: "rgba(255,255,255,0.7)" }}
              label={{ value: "Score", angle: -90, position: "insideLeft", fontSize: 10, fill: "rgba(255,255,255,0.5)" }}
            />
            <Tooltip content={<CustomBarTooltip />} cursor={{ fill: "rgba(255,255,255,0.05)" }} />
            <Bar dataKey="compositeScore" radius={[4, 4, 0, 0]}>
              {data.map((entry, index) => (
                <Cell
                  key={index}
                  fill={RAINBOW_COLORS[(entry.sessionIndex - 1) % RAINBOW_COLORS.length]}
                />
              ))}
            </Bar>
          </BarChart>
        </div>
      </div>
    </div>
  );
}

export default function StudentAnalyticsModal({
  open,
  onClose,
  studentId,
  studentName,
  classroomId,
}: StudentAnalyticsModalProps) {
  const [results, setResults] = useState<ResultEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const token = localStorage.getItem("token");

  useEffect(() => {
    if (!open || !studentId || !classroomId) return;

    const fetchResults = async () => {
      setLoading(true);
      try {
        const res = await fetch(
          `${API_BASE}/teacher/classroom/${classroomId}/student/${studentId}/results`,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        if (res.ok) {
          const data = await res.json();
          setResults(data);
        }
      } catch {
        console.error("Failed to fetch student results");
      } finally {
        setLoading(false);
      }
    };

    fetchResults();
  }, [open, studentId, classroomId]);

  const level1Data = buildBarData(results, 1);
  const level2Data = buildBarData(results, 2);

  return (
    <Dialog open={open} onOpenChange={(isOpen) => { if (!isOpen) onClose(); }}>
      <DialogContent className="max-w-2xl bg-[#1a1a2e] border-white/10 text-white max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="font-pixel text-lg text-white">
            {studentName} — Progress Analytics
          </DialogTitle>
        </DialogHeader>

        {loading ? (
          <div className="flex justify-center py-12">
            <Loader2 className="animate-spin text-white" size={32} />
          </div>
        ) : (
          <div className="mt-4">
            <ProgressChart data={level1Data} title="Level 1 — Letters" />
            <ProgressChart data={level2Data} title="Level 2 — Words" />

            {results.length > 0 && (
              <PixelCard variant="purple" className="text-white mt-4">
                <p className="font-pixel text-xs">
                  Total Graded Sessions: {countSessions([...level1Data, ...level2Data])}
                </p>
              </PixelCard>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
