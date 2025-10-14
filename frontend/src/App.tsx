import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Auth from "./pages/Auth";
import StudentDashboard from "./pages/student/Dashboard";
import Learn from "./pages/student/Learn";
import Play from "./pages/student/Play";
import PlaySession from "./pages/student/PlaySession";
import StudentSettings from "./pages/student/Settings";
import TeacherDashboard from "./pages/teacher/Dashboard";
import Classroom from "./pages/teacher/Classroom";
import TeacherSettings from "./pages/teacher/Settings";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Auth />} />

          {/* Student Routes */}
          <Route path="/student/dashboard" element={<StudentDashboard />} />
          <Route path="/student/learn" element={<Learn />} />
          <Route path="/student/play" element={<Play />} />
          <Route path="/student/play/session" element={<PlaySession />} />
          <Route path="/student/settings" element={<StudentSettings />} />

          {/* Teacher Routes */}
          <Route path="/teacher/dashboard" element={<TeacherDashboard />} />
          <Route path="/teacher/classroom" element={<Classroom />} />
          <Route path="/teacher/settings" element={<TeacherSettings />} />

          {/* 404 */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
