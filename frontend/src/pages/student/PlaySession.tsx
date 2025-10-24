import { useState, useEffect } from "react"; 
import { useNavigate, useSearchParams } from "react-router-dom"; 
import { Logo } from "@/components/Logo"; 
import { PixelButton } from "@/components/PixelButton"; 
import { PixelCard } from "@/components/PixelCard"; 
import { ArrowLeft } from "lucide-react"; 

// 🔹 Derive backend root dynamically from VITE_API_URL 
const BASE_URL = import.meta.env.VITE_API_URL.replace("/api/auth", ""); 

// 🔹 Detection API helpers 
async function startDetection() { 
  const res = await fetch(`${BASE_URL}/api/detect/start`, { method: "POST" }); 
  return res.json(); 
} 
async function stopDetection() { 
  const res = await fetch(`${BASE_URL}/api/detect/stop`, { method: "POST" }); 
  return res.json(); 
} 
async function getDetectionStatus() { 
  const res = await fetch(`${BASE_URL}/api/detect/status`); 
  return res.json(); 
} 

export default function PlaySession() { 
  const navigate = useNavigate(); 
  const [searchParams] = useSearchParams(); 
  const sessionType = searchParams.get("type") || "practice"; 
  const level = parseInt(searchParams.get("level") || "1"); 

  const [words, setWords] = useState<string[]>([]); 
  const [typedWords, setTypedWords] = useState<string[]>([]); 
  const [currentWordIndex, setCurrentWordIndex] = useState(0); 
  const [userInput, setUserInput] = useState(""); 
  const [activeKeys, setActiveKeys] = useState<{ [key: string]: string }>({}); 
  const [startTime] = useState(Date.now()); 
  const [wpm, setWpm] = useState(0); 
  const [accuracy, setAccuracy] = useState(100); 

  // 🔹 Finger detection state 
  const [detecting, setDetecting] = useState(false); 
  const [isCalibrating, setIsCalibrating] = useState(false); 
  const [calibrationDone, setCalibrationDone] = useState(false); 
  const [correctCount, setCorrectCount] = useState(0); 
  const [incorrectCount, setIncorrectCount] = useState(0); 
  const [lastKey, setLastKey] = useState<string | null>(null); 

  const keyboardLayout = [ 
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"], 
    ["A", "S", "D", "F", "G", "H", "J", "K", "L"], 
    ["Z", "X", "C", "V", "B", "N", "M"], 
  ]; 

  // ============================================================ 
  // 🧠 Send expected words depending on level type 
  // ============================================================ 
  const sendExpectedWords = async (level: number, words: string[]) => { 
    try { 
      let formattedWords: string[] = []; 

      if (level === 1) { 
        formattedWords = words.map((w) => w.trim()); 
      } else if (level === 2 || level === 3) { 
        formattedWords = words.map((w) => w.trim()); 
      } else { 
        formattedWords = [words.join(" ")]; 
      } 

      const payload = { words: formattedWords }; 
      console.log("📤 Sending expected words →", payload); 

      await fetch(`${BASE_URL}/api/detect/set-expected`, { 
        method: "POST", 
        headers: { "Content-Type": "application/json" }, 
        body: JSON.stringify(payload), 
      }); 
    } catch (error) { 
      console.error("❌ Failed to send expected words:", error); 
    } 
  }; 

  // ============================================================ 
  // Fetch words from backend + sync to Python detection 
  // ============================================================ 
  useEffect(() => { 
    const fetchTypingData = async () => { 
      try { 
        const res = await fetch(`http://localhost:5000/api/typing/level/${level}`); 
        const data = await res.json(); 
        if (data && data.data) { 
          const text = data.data.join(" "); 
          const wordArray = text.split(" "); 
          setWords(wordArray); 
          await sendExpectedWords(level, wordArray); 
        } 
      } catch (error) { 
        console.error("Error fetching typing data:", error); 
      } 
    }; 
    fetchTypingData(); 
  }, [level]); 

  // ============================================================ 
  // Handle typing input 
  // ============================================================ 
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => { 
    const value = e.target.value; 

    if (value.endsWith(" ")) { 
      const typedWord = value.trim(); 
      setTypedWords((prev) => { 
        const updated = [...prev]; 
        updated[currentWordIndex] = typedWord; 
        return updated; 
      }); 
      setCurrentWordIndex((prev) => prev + 1); 
      setUserInput(""); 
      return; 
    } 

    setUserInput(value); 
  }; 

  // ============================================================ 
  // Prevent backspace/delete & highlight pressed keys 
  // ============================================================ 
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => { 
    if (e.key === "Backspace" || e.key === "Delete") { 
      e.preventDefault(); 
      return; 
    } 
  }; 

  // ============================================================ 
  // Compute WPM & Accuracy dynamically 
  // ============================================================ 
  useEffect(() => { 
    const correctCount = typedWords.filter( 
      (typed, i) => typed && typed === words[i] 
    ).length; 
    const totalTyped = typedWords.filter(Boolean).length; 
    const accuracyVal = totalTyped > 0 ? Math.round((correctCount / totalTyped) * 100) : 100; 
    const timeElapsed = (Date.now() - startTime) / 1000 / 60; 
    setWpm(Math.round(correctCount / timeElapsed) || 0); 
    setAccuracy(accuracyVal); 
  }, [typedWords, startTime]); 

  // ============================================================ 
  // Start/Stop Detection Handlers 
  // ============================================================ 
  const handleStartDetection = async () => { 
    try { 
      setIsCalibrating(true); 
      setCalibrationDone(false); 
      await startDetection(); 
      setDetecting(true); 
      setTimeout(() => { 
        setIsCalibrating(false); 
        setCalibrationDone(true); 
        setTimeout(() => setCalibrationDone(false), 3000); 
      }, 10000); 
    } catch (err) { 
      console.error("Failed to start detection:", err); 
      setIsCalibrating(false); 
    } 
  }; 

  const handleStopDetection = async () => { 
    try { 
      await stopDetection(); 
      setDetecting(false); 
    } catch (err) { 
      console.error("Failed to stop detection:", err); 
    } 
  }; 

  // ============================================================ 
  // Auto Start/Stop Detection on Page Load 
  // ============================================================ 
  useEffect(() => { 
    handleStartDetection(); 
    return () => { 
      handleStopDetection(); 
    }; 
  }, []); 

  // ============================================================ 
  // 🔁 Realtime polling for Python detection correctness 
  // ============================================================ 
  useEffect(() => { 
    const interval = setInterval(async () => { 
      try { 
        const res = await getDetectionStatus(); 
        if (!res || !res.key) return; 
        const key = res.key.toUpperCase(); 
        const isCorrect = res.correct === true; 

        // Highlight pressed key with correct color 
        setActiveKeys((prev) => ({ 
          ...prev, 
          [key]: isCorrect ? "bg-green-500 text-white" : "bg-red-500 text-white", 
        })); 

        // Increment real-time counters 
        if (key !== lastKey) { 
          if (isCorrect) setCorrectCount((prev) => prev + 1); 
          else setIncorrectCount((prev) => prev + 1); 
          setLastKey(key); 
        } 

        // Auto-clear highlight after short delay 
        setTimeout(() => { 
          setActiveKeys((prev) => { 
            const updated = { ...prev }; 
            delete updated[key]; 
            return updated; 
          }); 
        }, 500); 
      } catch (err) { 
        // silent 
      } 
    }, 300); // faster refresh for real-time feel 
    return () => clearInterval(interval); 
  }, [lastKey]); 

  // ============================================================ 
  // Render words per letter 
  // ============================================================ 
  const renderWord = (word: string, index: number) => { 
    const typed = index === currentWordIndex ? userInput : typedWords[index] || ""; 
    const isCurrent = index === currentWordIndex; 

    return ( 
      <span key={index} className="mr-3"> 
        {word.split("").map((char, i) => { 
          let color = "text-muted-foreground"; 
          if (i < typed.length) { 
            color = typed[i] === char ? "text-green-600" : "text-red-600"; 
          } else if (isCurrent && i === typed.length) { 
            color = "text-orange-400 underline animate-pulse"; 
          } 
          return ( 
            <span key={i} className={`font-pixel ${color}`}> 
              {char} 
            </span> 
          ); 
        })} 
      </span> 
    ); 
  }; 

  const handleFinish = () => navigate("/student/play"); 
  const isFinished = currentWordIndex >= words.length; 

  // ============================================================ 
  // UI 
  // ============================================================ 
  return ( 
    <div className="min-h-screen p-8 flex flex-col items-center"> 
      {/* 🟡 Calibration Popup */} 
      {(isCalibrating || calibrationDone) && ( 
        <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-50"> 
          <PixelCard className="p-8 text-center"> 
            {isCalibrating ? ( 
              <> 
                <p className="font-pixel text-lg text-yellow-400 mb-2"> 
                  🔧 Calibrating Keyboard Layout... 
                </p> 
                <p className="font-pixel text-sm text-muted-foreground"> 
                  Please remove your hands from the keyboard. 
                </p> 
              </> 
            ) : ( 
              <p className="font-pixel text-lg text-green-500">✅ Calibration Complete!</p> 
            )} 
          </PixelCard> 
        </div> 
      )} 

      <div className="max-w-7xl w-full relative"> 
        {/* Header */} 
        <div className="flex items-center justify-between mb-8"> 
          <div className="flex items-center gap-4"> 
            <PixelButton variant="secondary" onClick={() => navigate("/student/play")}> 
              <ArrowLeft size={20} /> 
            </PixelButton> 
            <Logo /> 
          </div> 
          <div className="font-pixel text-sm"> 
            {sessionType === "evaluated" ? "🏆 Graded Session" : "🎮 Practice Mode"} - Level{" "} 
            {level} 
          </div> 
        </div> 

        {/* Stats */} 
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8"> 
          <PixelCard variant="yellow"> 
            <p className="font-pixel text-xs mb-1">WPM</p> 
            <p className="font-pixel text-2xl">{wpm}</p> 
          </PixelCard> 
          <PixelCard variant="orange"> 
            <p className="font-pixel text-xs mb-1">Accuracy</p> 
            <p className="font-pixel text-2xl">{accuracy}%</p> 
          </PixelCard> 
          <PixelCard variant="green"> 
            <p className="font-pixel text-xs mb-1">Correct (Finger)</p> 
            <p className="font-pixel text-2xl">{correctCount}</p> 
          </PixelCard> 
          <PixelCard variant="red"> 
            <p className="font-pixel text-xs mb-1">Incorrect (Finger)</p> 
            <p className="font-pixel text-2xl">{incorrectCount}</p> 
          </PixelCard> 
        </div> 

        {/* Typing Area */} 
        {!isFinished ? ( 
          <PixelCard className="mb-8 flex flex-col items-center justify-center text-center py-8"> 
            <div className="font-pixel text-lg mb-6 flex flex-wrap justify-center gap-2 max-w-3xl leading-relaxed"> 
              {words.map((word, index) => renderWord(word, index))} 
            </div> 

            <input 
              type="text" 
              value={userInput} 
              onChange={handleChange} 
              onKeyDown={handleKeyDown} 
              className="text-center w-2/3 md:w-1/2 px-4 py-3 bg-input border-[3px] border-border text-foreground font-pixel text-lg focus:outline-none focus:ring-2 focus:ring-primary rounded-md" 
              placeholder="Type here..." 
              autoFocus 
            /> 

            {/* Keyboard */} 
            <div className="font-pixel text-sm text-muted-foreground mt-6 mb-2"> 
              Keyboard (60% Layout) 
            </div> 
            <div className="flex flex-col items-center gap-2"> 
              {keyboardLayout.map((row, rowIdx) => ( 
                <div key={rowIdx} className="flex gap-2 justify-center"> 
                  {row.map((key) => ( 
                    <div 
                      key={key} 
                      className={`pixel-border w-10 h-10 flex items-center justify-center font-pixel text-sm border border-border rounded-md transition-colors duration-150 ${ 
                        activeKeys[key] 
                          ? activeKeys[key] 
                          : "bg-muted text-foreground" 
                      }`} 
                    > 
                      {key} 
                    </div> 
                  ))} 
                </div> 
              ))} 
              <div 
                className={`pixel-border w-[300px] h-10 flex items-center justify-center font-pixel text-sm border border-border rounded-md mt-2 ${ 
                  activeKeys[" "] ? activeKeys[" "] : "bg-muted text-foreground" 
                }`} 
              > 
                SPACE 
              </div> 
            </div> 
          </PixelCard> 
        ) : ( 
          <PixelCard className="text-center py-8"> 
            <p className="font-pixel text-xl text-green-600 mb-4">🎉 Session Complete!</p> 
            <p className="font-pixel text-sm text-muted-foreground"> 
              You typed {typedWords.length} words with {accuracy}% accuracy. 
            </p> 
          </PixelCard> 
        )} 

        {/* Detection Controls */} 
        <div className="flex justify-center mt-6 gap-4"> 
          {!detecting ? ( 
            <PixelButton variant="orange" size="lg" onClick={handleStartDetection}> 
              🎥 Start Detection 
            </PixelButton> 
          ) : ( 
            <PixelButton variant="red" size="lg" onClick={handleStopDetection}> 
              🛑 Stop Detection 
            </PixelButton> 
          )} 
          <PixelButton variant="primary" size="lg" onClick={handleFinish}> 
            {isFinished ? "Back to Play" : "Finish Session"} 
          </PixelButton> 
        </div> 
      </div> 
    </div> 
  ); 
}
