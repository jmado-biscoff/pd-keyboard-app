import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Logo } from "@/components/Logo";
import { PixelButton } from "@/components/PixelButton";
import { PixelCard } from "@/components/PixelCard";
import { ArrowLeft } from "lucide-react";
import bgImage from "@/assets/b6.jpg";

export default function FullKeyboardModule() {
  const navigate = useNavigate();

  const lessons = [
    "The quick brown fox jumps over the lazy dog",
    "Practice makes perfect",
    "Typing fast is fun and useful",
    "Keep your eyes on the screen",
    "Accuracy first, speed later",
    "Use all your fingers equally",
    "Typing is a lifelong skill",
    "Stay calm and keep typing",
    "Every keystroke builds muscle memory",
    "Now you know the full keyboard layout",
  ];

  const [currentIndex, setCurrentIndex] = useState(0);
  const [userInput, setUserInput] = useState("");
  const [wpm, setWpm] = useState(0);
  const [accuracy, setAccuracy] = useState(100);
  const [completed, setCompleted] = useState(false);
  const [lessonStart, setLessonStart] = useState(Date.now());
  const [elapsedTime, setElapsedTime] = useState(0);
  const [timerRunning, setTimerRunning] = useState(true);

  useEffect(() => {
    let interval: any;
    if (timerRunning && !completed) {
      interval = setInterval(() => setElapsedTime(Math.floor((Date.now() - lessonStart) / 1000)), 1000);
    }
    return () => clearInterval(interval);
  }, [timerRunning, completed, lessonStart]);

  const currentText = lessons[currentIndex];

  useEffect(() => {
    const elapsedMinutes = Math.max((Date.now() - lessonStart) / 1000 / 60, 0.01);
    const wordsTyped = userInput.trim().split(" ").length;
    setWpm(Math.round(wordsTyped / elapsedMinutes));

    let correct = 0;
    for (let i = 0; i < userInput.length; i++)
      if (userInput[i] === currentText[i]) correct++;
    setAccuracy(userInput.length > 0 ? Math.round((correct / userInput.length) * 100) : 100);
  }, [userInput, lessonStart, currentText]);

  useEffect(() => {
    if (userInput.length >= currentText.length && !completed) setTimeout(() => handleNext(), 400);
  }, [userInput, currentText]);

  const handleTyping = (val: string) => setUserInput(val);
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Backspace" || e.key === "Delete") e.preventDefault();
  };

  const handleNext = () => {
    if (currentIndex < lessons.length - 1) {
      setCurrentIndex((p) => p + 1);
      setUserInput("");
      setLessonStart(Date.now());
      setElapsedTime(0);
    } else {
      setCompleted(true);
      setTimerRunning(false);
    }
  };

  const handleFinish = () => {
    const saved = JSON.parse(localStorage.getItem("typingModuleProgress") || "{}");
    saved[6] = { completed: true };
    localStorage.setItem("typingModuleProgress", JSON.stringify(saved));
    navigate("/student/learn");
  };

  return (
    <div className="relative min-h-screen overflow-hidden"
      style={{backgroundImage:`url(${bgImage})`,backgroundSize:"cover",backgroundPosition:"center"}}>
      <div className="absolute inset-0 bg-black/30 z-0" />
      <div className="relative z-10 p-8 min-h-screen text-white">
        <div className="max-w-6xl mx-auto">
          <div className="flex items-center justify-between mb-8">
            <div className="flex items-center gap-4">
              <PixelButton variant="secondary" onClick={()=>navigate("/student/learn")}><ArrowLeft size={20}/></PixelButton>
              <Logo />
            </div>
            <h1 className="font-pixel text-xl text-black">Module 6: Full Keyboard Practice</h1>
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            <PixelCard variant="orange"><p className="font-pixel text-xs mb-1">WPM</p><p className="font-pixel text-2xl">{wpm}</p></PixelCard>
            <PixelCard variant="purple"><p className="font-pixel text-xs mb-1">Accuracy</p><p className="font-pixel text-2xl">{accuracy}%</p></PixelCard>
            <PixelCard variant="green"><p className="font-pixel text-xs mb-1">Timer</p><p className="font-pixel text-2xl">{elapsedTime}s</p></PixelCard>
            <PixelCard variant="yellow"><p className="font-pixel text-xs mb-1">Lesson</p><p className="font-pixel text-2xl">{currentIndex+1}/10</p></PixelCard>
          </div>

          <PixelCard className="mb-8 bg-black/60 border-[3px] border-yellow-300 backdrop-blur-sm text-center">
            <div className="font-pixel text-lg mb-4">
              {currentText.split("").map((ch,i)=>{
                let c="text-gray-400";
                if(i<userInput.length) c=userInput[i]===ch?"text-green-400":"text-red-500";
                else if(i===userInput.length&&!completed) c="text-purple-400 animate-pulse";
                return <span key={i} className={c}>{ch}</span>;
              })}
            </div>
            <input type="text" value={userInput} onChange={(e)=>handleTyping(e.target.value)} onKeyDown={handleKeyDown} disabled={completed}
              className={`w-full px-4 py-3 font-pixel text-sm border-[3px] ${completed?"bg-gray-700 border-gray-500 text-gray-400":"bg-black/70 border-yellow-300 text-white"}`} placeholder={completed?"Module completed!":"Type the sentence above..."} autoFocus/>
          </PixelCard>

          {completed&&(
            <div className="text-center">
              <PixelCard className="inline-block p-8 bg-black/70 border-[3px] border-yellow-300 backdrop-blur-md">
                <h2 className="font-pixel text-2xl mb-4 text-yellow-200">🎉 Module Complete!</h2>
                <p className="font-pixel mb-4">{accuracy>=95?"Outstanding typing!":accuracy>=85?"Very good performance!":"Try to slow down for accuracy."}</p>
                <div className="grid grid-cols-2 gap-4 mb-6">
                  <PixelCard className="bg-black/50 border border-yellow-200 text-center"><p className="font-pixel text-xs mb-1 text-yellow-200">WPM</p><p className="font-pixel text-2xl">{wpm}</p></PixelCard>
                  <PixelCard className="bg-black/50 border border-yellow-200 text-center"><p className="font-pixel text-xs mb-1 text-yellow-200">Accuracy</p><p className="font-pixel text-2xl">{accuracy}%</p></PixelCard>
                </div>
                <PixelButton variant="primary" size="lg" onClick={handleFinish}>Return to Modules</PixelButton>
              </PixelCard>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
