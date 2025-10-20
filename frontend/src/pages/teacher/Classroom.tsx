import { useNavigate } from "react-router-dom";
import { useState, useEffect } from "react";
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
} from "lucide-react";
import { toast } from "sonner";

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

  // 🔹 Fetch classrooms from backend
  const fetchClassrooms = async () => {
    try {
      const res = await fetch(
        `${import.meta.env.VITE_API_URL}/teacher/my-classrooms`,
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

  // 🔹 Create new classroom
  const handleCreate = async () => {
    if (!newClassName.trim()) return toast.error("Enter a classroom name");
    try {
      const res = await fetch(
        `${import.meta.env.VITE_API_URL}/teacher/create-classroom`,
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
          `${
            import.meta.env.VITE_API_URL
          }/teacher/classroom/${classroomId}/students`,
          {
            headers: { Authorization: `Bearer ${token}` },
          }
        ),
        fetch(
          `${
            import.meta.env.VITE_API_URL
          }/teacher/classroom/${classroomId}/requests`,
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
        `${
          import.meta.env.VITE_API_URL
        }/teacher/classroom/${classId}/${decision}`,
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
    <div className="min-h-screen p-8 bg-gradient-to-br from-[#ffb067]/10 to-[#ffa94d]/20">
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
        <PixelCard variant="orange" className="mb-10 text-white">
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
            <PixelButton onClick={handleCreate}>Create</PixelButton>
          </div>
        </PixelCard>

        {/* 🏫 Classroom Cards */}
        <div className="space-y-6">
          {classrooms.map((cls) => (
            <PixelCard
              key={cls._id}
              variant="orange"
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
                    variant="purple"
                    onClick={() => toggleExpand(cls._id)}
                    className="hover:scale-110 transition-transform"
                  >
                    <Users size={16} />
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
                              className="font-pixel text-xs flex justify-between items-center bg-white/10 p-2 rounded-md"
                            >
                              <div className="flex items-center gap-2">
                                <span className="text-lg">
                                  {["🦁", "🐯", "🐻", "🦊", "🐺"][idx % 5]}
                                </span>
                                <span>{student.name || "Unnamed Student"}</span>
                              </div>
                              <span className="opacity-70 text-[10px]">
                                {student.email}
                              </span>
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <p className="font-pixel text-xs text-center opacity-70 py-2">
                          📭 No students joined yet.
                        </p>
                      )}
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
  );
}
