import { useNavigate } from "react-router-dom";
import { useState, useEffect, useRef } from "react";
import { Logo } from "@/components/Logo";
import { PixelButton } from "@/components/PixelButton";
import { PixelCard } from "@/components/PixelCard";
import {
  ArrowLeft,
  Copy,
  PlusCircle,
  Users,
  School,
  Loader2,
  CheckCircle,
  XCircle,
  Settings,
  Clock,
  BarChart3,
  Trash2,
} from "lucide-react";
import { toast } from "sonner";
import StudentAnalyticsModal from "@/components/teacher/StudentAnalyticsModal";
import bgGif from "@/assets/classroom_background.gif";
import { resolveProfileImage } from "@/utils/profileAssets";
import fallbackProfilePic from "@/assets/cat-profile.jpg";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:5000/api";

export default function Classroom() {
  const navigate = useNavigate();
  const token = localStorage.getItem("token");

  const [classrooms, setClassrooms] = useState<any[]>([]);
  const [newClassName, setNewClassName] = useState("");
  const [expanded, setExpanded] = useState<string | null>(null);
  const [studentsMap, setStudentsMap] = useState<Record<string, any[]>>({});
  const [pendingRequestsMap, setPendingRequestsMap] = useState<
    Record<string, any[]>
  >({});
  const [loadingStudents, setLoadingStudents] = useState(false);

  // Evaluation settings state
  const [evalSettingsMap, setEvalSettingsMap] = useState<
    Record<string, { level: string; proctorTimerMinutes: number; isActive: boolean; activatedAt: string | null }>
  >({});
  const [evalCountdowns, setEvalCountdowns] = useState<Record<string, number>>({});
  const countdownRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Delete confirmation state
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);

  // Student analytics modal state
  const [analyticsModal, setAnalyticsModal] = useState<{
    open: boolean;
    studentId: string;
    studentName: string;
    classroomId: string;
  }>({ open: false, studentId: "", studentName: "", classroomId: "" });

  // 🔹 Fetch classrooms from backend
  const fetchClassrooms = async () => {
    try {
      const res = await fetch(
        `${API_BASE}/teacher/my-classrooms`,
        {
          headers: { Authorization: `Bearer ${token}` },
        }
      );
      const data = await res.json();
      if (res.ok) setClassrooms(data);
      else toast.error(data.message || "Failed to fetch classrooms");
    } catch {
      toast.error("Error fetching classrooms");
    }
  };

  useEffect(() => {
    fetchClassrooms();
  }, []);

  // Initialize eval settings from classroom data
  useEffect(() => {
    classrooms.forEach((cls) => {
      if (cls.activeEvaluation && !evalSettingsMap[cls._id]) {
        setEvalSettingsMap((prev) => ({
          ...prev,
          [cls._id]: {
            level: cls.activeEvaluation.level || "characters",
            proctorTimerMinutes: cls.activeEvaluation.proctorTimerMinutes || 30,
            isActive: cls.activeEvaluation.isActive || false,
            activatedAt: cls.activeEvaluation.activatedAt || null,
          },
        }));
      }
    });
  }, [classrooms]);

  // Countdown timer for active evaluations
  useEffect(() => {
    if (countdownRef.current) clearInterval(countdownRef.current);
    countdownRef.current = setInterval(() => {
      setEvalCountdowns((prev) => {
        const next: Record<string, number> = {};
        for (const [id, settings] of Object.entries(evalSettingsMap)) {
          if (settings.isActive && settings.activatedAt) {
            const elapsed = (Date.now() - new Date(settings.activatedAt).getTime()) / 1000;
            const total = settings.proctorTimerMinutes * 60;
            const remaining = Math.max(0, Math.floor(total - elapsed));
            next[id] = remaining;
            if (remaining <= 0) {
              setEvalSettingsMap((p) => ({ ...p, [id]: { ...p[id], isActive: false, activatedAt: null } }));
            }
          }
        }
        return next;
      });
    }, 1000);
    return () => { if (countdownRef.current) clearInterval(countdownRef.current); };
  }, [evalSettingsMap]);

  // Save evaluation settings
  const saveEvalSettings = async (classroomId: string) => {
    const settings = evalSettingsMap[classroomId];
    if (!settings) return;
    try {
      const res = await fetch(
        `${API_BASE}/teacher/classroom/${classroomId}/evaluation`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
          body: JSON.stringify({ level: settings.level, proctorTimerMinutes: settings.proctorTimerMinutes }),
        }
      );
      if (res.ok) toast.success("Settings saved");
      else toast.error("Failed to save settings");
    } catch { toast.error("Error saving settings"); }
  };

  // Toggle evaluation activation
  const toggleEvaluation = async (classroomId: string) => {
    const settings = evalSettingsMap[classroomId];
    const activate = !settings?.isActive;
    try {
      // Save settings first if activating
      if (activate) await saveEvalSettings(classroomId);
      const res = await fetch(
        `${API_BASE}/teacher/classroom/${classroomId}/evaluation/activate`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
          body: JSON.stringify({ activate }),
        }
      );
      const data = await res.json();
      if (res.ok) {
        toast.success(activate ? "Activity activated!" : "Activity deactivated");
        setEvalSettingsMap((prev) => ({
          ...prev,
          [classroomId]: {
            ...prev[classroomId],
            isActive: data.activeEvaluation.isActive,
            activatedAt: data.activeEvaluation.activatedAt,
          },
        }));
      } else toast.error(data.message || "Failed to toggle activity");
    } catch { toast.error("Error toggling activity"); }
  };

  const formatCountdown = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  // 🔹 Delete classroom
  const handleDelete = async (classroomId: string) => {
    try {
      const res = await fetch(
        `${API_BASE}/teacher/classroom/${classroomId}`,
        {
          method: "DELETE",
          headers: { Authorization: `Bearer ${token}` },
        }
      );
      const data = await res.json();
      if (res.ok) {
        toast.success("Classroom deleted!");
        setClassrooms((prev) => prev.filter((c) => c._id !== classroomId));
        setDeleteConfirm(null);
        if (expanded === classroomId) setExpanded(null);
      } else toast.error(data.message || "Failed to delete classroom");
    } catch {
      toast.error("Error deleting classroom");
    }
  };

  // 🔹 Create new classroom
  const handleCreate = async () => {
    if (!newClassName.trim()) return toast.error("Enter a classroom name");
    try {
      const res = await fetch(
        `${API_BASE}/teacher/create-classroom`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({ name: newClassName }),
        }
      );
      const data = await res.json();
      if (res.ok) {
        toast.success("Classroom created!");
        setClassrooms((prev) => [...prev, data.classroom]);
        setNewClassName("");
      } else toast.error(data.message || "Failed to create classroom");
    } catch {
      toast.error("Error creating classroom");
    }
  };

  // 🔹 Copy classroom code
  const copyClassroomCode = (code: string) => {
    navigator.clipboard.writeText(code);
    toast.success("Classroom code copied!");
  };

  // 🔹 Fetch students and pending requests for a classroom
  const fetchClassroomDetails = async (classroomId: string) => {
    try {
      setLoadingStudents(true);

      const [studentsRes, requestsRes] = await Promise.all([
        fetch(
          `${API_BASE}/teacher/classroom/${classroomId}/students`,
          {
            headers: { Authorization: `Bearer ${token}` },
          }
        ),
        fetch(
          `${API_BASE}/teacher/classroom/${classroomId}/requests`,
          {
            headers: { Authorization: `Bearer ${token}` },
          }
        ),
      ]);

      const studentsData = await studentsRes.json();
      const requestsData = await requestsRes.json();

      if (studentsRes.ok)
        setStudentsMap((prev) => ({
          ...prev,
          [classroomId]: studentsData.students,
        }));
      else toast.error(studentsData.message || "Failed to load students");

      if (requestsRes.ok)
        setPendingRequestsMap((prev) => ({
          ...prev,
          [classroomId]: requestsData,
        }));
      else toast.error(requestsData.message || "Failed to load join requests");
    } catch {
      toast.error("Error loading classroom details");
    } finally {
      setLoadingStudents(false);
    }
  };

  const toggleExpand = (classroomId: string) => {
    if (expanded === classroomId) {
      setExpanded(null);
    } else {
      setExpanded(classroomId);
      fetchClassroomDetails(classroomId);
    }
  };

  // 🔹 Handle approve/reject
  const handleDecision = async (
    classId: string,
    studentId: string,
    decision: "approve" | "reject"
  ) => {
    try {
      const res = await fetch(
        `${API_BASE}/teacher/classroom/${classId}/${decision}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({ studentId }),
        }
      );

      const data = await res.json();
      if (res.ok) {
        toast.success(data.message);
        fetchClassroomDetails(classId); // refresh
      } else toast.error(data.message);
    } catch {
      toast.error("Error processing request");
    }
  };

  return (
    <div className="relative min-h-screen overflow-hidden">
      {/* Background GIF */}
      <img
        src={bgGif}
        alt="Background"
        className="absolute top-0 left-0 w-full h-full object-cover -z-10"
      />

      {/* Page Content */}
      <div className="relative z-10 p-8 bg-black/20 min-h-screen">
        <div className="max-w-6xl mx-auto">
          {/* 🔙 Back to Dashboard */}
          <div className="flex items-center gap-4 mb-12">
            <PixelButton
              variant="secondary"
              onClick={() => navigate("/teacher/dashboard")}
            >
              <ArrowLeft size={20} />
            </PixelButton>
            <Logo />
          </div>

          {/* 🔸 Create new classroom */}
          <PixelCard variant="black" className="mb-10 text-white">
            <h2 className="font-pixel text-xl mb-3 flex items-center gap-2">
              <PlusCircle size={22} /> Create New Classroom
            </h2>
            <div className="flex gap-3">
              <input
                type="text"
                placeholder="Enter classroom name"
                className="px-4 py-2 rounded-lg text-black flex-1 font-pixel text-sm"
                value={newClassName}
                onChange={(e) => setNewClassName(e.target.value)}
              />
              <PixelButton variant="green" onClick={handleCreate}>Create</PixelButton>
            </div>
          </PixelCard>

          {/* 🏫 Classroom Cards */}
          <div className="space-y-6">
            {classrooms.map((cls) => (
              <PixelCard
                key={cls._id}
                variant="black"
                className="text-white shadow-lg transition-all hover:scale-[1.01] border-2 border-transparent hover:border-white/20"
              >
                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-3">
                    <School size={24} />
                    <div>
                      <p className="font-pixel text-sm">{cls.name}</p>
                      <p className="font-pixel text-xl opacity-80">{cls.code}</p>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <PixelButton
                      variant="accent"
                      onClick={() => copyClassroomCode(cls.code)}
                      className="hover:scale-110 transition-transform"
                    >
                      <Copy size={16} />
                    </PixelButton>
                    <PixelButton
                      variant="blue"
                      onClick={() => toggleExpand(cls._id)}
                      className="hover:scale-110 transition-transform"
                    >
                      {expanded === cls._id ? "Hide Students" : "Show Students"}
                    </PixelButton>
                    <PixelButton
                      variant="red"
                      onClick={() => setDeleteConfirm(cls._id)}
                      className="hover:scale-110 transition-transform"
                    >
                      <Trash2 size={16} />
                    </PixelButton>
                  </div>
                </div>

                {/* Expanded student + pending requests list */}
                {expanded === cls._id && (
                  <div className="mt-4 bg-black/20 rounded-lg p-4 overflow-hidden">
                    {loadingStudents ? (
                      <div className="flex justify-center py-4">
                        <Loader2 className="animate-spin" />
                      </div>
                    ) : (
                      <>
                        {/* 🧾 Pending Join Requests */}
                        <h3 className="font-pixel text-sm mb-3 flex items-center gap-2">
                          <Users size={16} /> Pending Join Requests
                        </h3>
                        {pendingRequestsMap[cls._id]?.length > 0 ? (
                          <ul className="space-y-2 mb-4">
                            {pendingRequestsMap[cls._id].map((student) => (
                              <li
                                key={student._id}
                                className="font-pixel text-xs flex justify-between items-center bg-white/10 p-2 rounded-md"
                              >
                                <div className="flex flex-col">
                                  <span>{student.name || "Unnamed Student"}</span>
                                  <span className="opacity-70 text-[10px]">
                                    {student.email}
                                  </span>
                                </div>
                                <div className="flex gap-2">
                                  <PixelButton
                                    variant="accent"
                                    onClick={() =>
                                      handleDecision(
                                        cls._id,
                                        student._id,
                                        "approve"
                                      )
                                    }
                                  >
                                    <CheckCircle size={14} />
                                  </PixelButton>
                                  <PixelButton
                                    variant="red"
                                    onClick={() =>
                                      handleDecision(
                                        cls._id,
                                        student._id,
                                        "reject"
                                      )
                                    }
                                  >
                                    <XCircle size={14} />
                                  </PixelButton>
                                </div>
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <p className="font-pixel text-xs text-center opacity-70 py-2">
                            No pending join requests.
                          </p>
                        )}

                        {/* 👩‍🎓 Enrolled Students */}
                        <h3 className="font-pixel text-sm mb-3 flex items-center gap-2 mt-4">
                          <Users size={16} /> Enrolled Students
                        </h3>
                        {studentsMap[cls._id]?.length > 0 ? (
                          <ul className="space-y-2">
                            {studentsMap[cls._id].map((student, idx) => (
                              <li
                                key={student._id}
                                className="font-pixel text-xs flex justify-between items-center bg-white/10 p-2 rounded-md cursor-pointer hover:bg-white/20 transition-colors"
                                onClick={() =>
                                  setAnalyticsModal({
                                    open: true,
                                    studentId: student._id,
                                    studentName: student.name || "Unnamed Student",
                                    classroomId: cls._id,
                                  })
                                }
                              >
                                <div className="flex items-center gap-2">
                                  <img
                                    src={resolveProfileImage(student.profilePicture) || fallbackProfilePic}
                                    alt="Avatar"
                                    className="w-8 h-8 rounded-md border border-white/20 object-cover image-render-pixel"
                                  />
                                  <span>{student.name || "Unnamed Student"}</span>
                                </div>
                                <div className="flex items-center gap-2">
                                  <span className="opacity-70 text-[10px]">
                                    {student.email}
                                  </span>
                                  <BarChart3 size={14} className="opacity-60" />
                                </div>
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <p className="font-pixel text-xs text-center opacity-70 py-2">
                            📭 No students joined yet.
                          </p>
                        )}

                        {/* Activity Settings Panel */}
                        <div className="mt-6 border-t border-white/20 pt-4">
                          <h3 className="font-pixel text-sm mb-3 flex items-center gap-2">
                            <Settings size={16} /> Activity Settings
                          </h3>

                          {/* Level Selector */}
                          <div className="mb-3">
                            <p className="font-pixel text-[10px] opacity-70 mb-2">LEVEL</p>
                            <div className="flex gap-2">
                              {(["characters", "words", "both"] as const).map((lvl) => {
                                const currentLevel = evalSettingsMap[cls._id]?.level || "characters";
                                const isBoth = currentLevel === "both";
                                const isSelected = lvl === "both" ? isBoth : (currentLevel === lvl || isBoth);

                                return (
                                  <PixelButton
                                    key={lvl}
                                    size="sm"
                                    variant={isSelected ? "green" : "blue"}
                                    onClick={() =>
                                      setEvalSettingsMap((prev) => ({
                                        ...prev,
                                        [cls._id]: { ...prev[cls._id], level: lvl },
                                      }))
                                    }
                                    disabled={evalSettingsMap[cls._id]?.isActive || (lvl === "both" && isBoth)}
                                    className={lvl === "both" && isBoth ? "opacity-30 grayscale" : ""}
                                  >
                                    {lvl === "characters" ? "Letters" : lvl.charAt(0).toUpperCase() + lvl.slice(1)}
                                  </PixelButton>
                                );
                              })}
                            </div>
                          </div>

                          {/* Session Open Duration */}
                          <div className="mb-3">
                            <p className="font-pixel text-[10px] opacity-70 mb-2">
                              <Clock size={12} className="inline mr-1" />
                              SESSION OPEN DURATION:
                            </p>
                            <div className="flex items-center gap-2">
                              <input
                                type="number"
                                min={5}
                                max={60}
                                value={evalSettingsMap[cls._id]?.proctorTimerMinutes || 30}
                                onChange={(e) => {
                                  const val = Math.min(60, Math.max(5, Number(e.target.value) || 5));
                                  setEvalSettingsMap((prev) => ({
                                    ...prev,
                                    [cls._id]: { ...prev[cls._id], proctorTimerMinutes: val },
                                  }));
                                }}
                                disabled={evalSettingsMap[cls._id]?.isActive}
                                className="w-20 px-2 py-1 rounded-md font-pixel text-sm text-black bg-white/90 disabled:opacity-50 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                              />
                              <span className="font-pixel text-xs opacity-70">minutes</span>
                            </div>
                            <p className="font-pixel text-[9px] opacity-50 mt-1">
                              Min: 5 minutes | Max: 60 minutes
                            </p>
                          </div>

                          {/* Activate/Deactivate Toggle */}
                          <div className="flex items-center gap-3">
                            <PixelButton
                              variant={evalSettingsMap[cls._id]?.isActive ? "red" : "green"}
                              onClick={() => toggleEvaluation(cls._id)}
                            >
                              {evalSettingsMap[cls._id]?.isActive
                                ? "Deactivate"
                                : "Activate Activity"}
                            </PixelButton>

                            {evalSettingsMap[cls._id]?.isActive && evalCountdowns[cls._id] !== undefined && (
                              <span className="font-pixel text-sm animate-pulse">
                                {formatCountdown(evalCountdowns[cls._id])} remaining
                              </span>
                            )}
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                )}
              </PixelCard>
            ))}
          </div>

          {classrooms.length === 0 && (
            <p className="text-center text-muted-foreground mt-10">
              No classrooms yet. Create one above!
            </p>
          )}
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      {deleteConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <PixelCard variant="red" className="text-white max-w-sm w-full mx-4">
            <h3 className="font-pixel text-lg mb-3">Delete Classroom?</h3>
            <p className="font-pixel text-xs opacity-80 mb-6">
              This will permanently remove the classroom and unenroll all students. This action cannot be undone.
            </p>
            <div className="flex gap-3 justify-end">
              <PixelButton variant="secondary" onClick={() => setDeleteConfirm(null)}>
                Cancel
              </PixelButton>
              <PixelButton variant="red" onClick={() => handleDelete(deleteConfirm)}>
                Delete
              </PixelButton>
            </div>
          </PixelCard>
        </div>
      )}

      {/* Student Analytics Modal */}
      <StudentAnalyticsModal
        open={analyticsModal.open}
        onClose={() => setAnalyticsModal((prev) => ({ ...prev, open: false }))}
        studentId={analyticsModal.studentId}
        studentName={analyticsModal.studentName}
        classroomId={analyticsModal.classroomId}
      />
    </div>
  );
}
