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
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

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
}

interface ChartPoint {
  session: number;
  date: string;
  compositeScore: number;
  grade: string;
  netWpm: number;
  accuracy: number;
  errorRate: number;
  sessionType: "practice" | "evaluated";
}

function ordinal(n: number): string {
  const s = ["th", "st", "nd", "rd"];
  const v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}

const CustomTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null;
  const data = payload[0].payload as ChartPoint;
  return (
    <div className="bg-black/90 border border-white/20 p-3 rounded font-pixel text-xs text-white">
      <p className="text-yellow-400 mb-1">{ordinal(data.session)} Trial</p>
      <p>Grade: {data.grade || "N/A"}</p>
      <p>Net WPM: {Math.round(data.netWpm)}</p>
      <p>Accuracy: {data.accuracy?.toFixed(1)}%</p>
      <p>Error Rate: {data.errorRate?.toFixed(1)}%</p>
      <p>Date: {data.date}</p>
    </div>
  );
};

function buildChartData(results: ResultEntry[], targetLevel: number): ChartPoint[] {
  return results
    .filter((r) => r.level === targetLevel)
    .map((r, idx) => ({
      session: idx + 1,
      date: new Date(r.createdAt).toLocaleDateString(),
      compositeScore: r.compositeScore || 0,
      grade: r.grade || "N/A",
      netWpm: r.netWpm || 0,
      accuracy: r.accuracy || 0,
      errorRate: r.errorRate || 0,
      sessionType: r.sessionType,
    }));
}

function ProgressChart({ data, title }: { data: ChartPoint[]; title: string }) {
  // Only show evaluated sessions (practice isn't saved to DB)
  const evaluatedData = data.filter((d) => d.sessionType === "evaluated");
  const merged = evaluatedData.map((d, idx) => ({
    ...d,
    session: idx + 1,
  }));

  if (merged.length === 0) {
    return (
      <div className="mb-6">
        <h4 className="font-pixel text-sm mb-2 text-white">{title}</h4>
        <div className="bg-white/5 rounded-lg p-6 text-center">
          <p className="font-pixel text-xs opacity-50 text-white">No sessions recorded yet</p>
        </div>
      </div>
    );
  }

  return (
    <div className="mb-6">
      <h4 className="font-pixel text-sm mb-2 text-white">{title}</h4>
      <div className="bg-white/5 rounded-lg p-3">
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={merged} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis
              dataKey="session"
              stroke="rgba(255,255,255,0.5)"
              tick={{ fontSize: 10, fontFamily: "monospace", fill: "rgba(255,255,255,0.7)" }}
              label={{ value: "Session", position: "insideBottom", offset: -2, fontSize: 10, fill: "rgba(255,255,255,0.5)" }}
            />
            <YAxis
              domain={[0, 100]}
              stroke="rgba(255,255,255,0.5)"
              tick={{ fontSize: 10, fontFamily: "monospace", fill: "rgba(255,255,255,0.7)" }}
              label={{ value: "Score", angle: -90, position: "insideLeft", fontSize: 10, fill: "rgba(255,255,255,0.5)" }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Line
              type="monotone"
              dataKey="compositeScore"
              name="Evaluated"
              stroke="#F4A942"
              strokeWidth={2}
              dot={{ r: 4, fill: "#F4A942" }}
              connectNulls
            />
          </LineChart>
        </ResponsiveContainer>
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
          `${import.meta.env.VITE_API_URL}/teacher/classroom/${classroomId}/student/${studentId}/results`,
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

  const level1Data = buildChartData(results, 1);
  const level2Data = buildChartData(results, 2);

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
            <ProgressChart data={level1Data} title="Level 1 — Characters" />
            <ProgressChart data={level2Data} title="Level 2 — Words" />

            {results.length > 0 && (
              <PixelCard variant="purple" className="text-white mt-4">
                <p className="font-pixel text-xs">
                  Total Evaluated Sessions: {results.filter((r) => r.sessionType === "evaluated").length}
                </p>
              </PixelCard>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
