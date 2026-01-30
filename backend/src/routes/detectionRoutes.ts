import express, { Request, Response } from "express";
import { spawn } from "child_process";
import path from "path";
import fs from "fs";

const router = express.Router();

// ============================================================
// 🔹 GLOBAL VARIABLES
// ============================================================

let pythonProcess: any = null;
let latestDetection: any = { key: "", finger: "", hand: "", correct: null };
let latestFrame: string | null = null;
let calibrationState: {
  done: boolean;
  detected: number;
  required: number;
  detected_keys: string[];
  frame?: string | null;
} = { done: false, detected: 0, required: 26, detected_keys: [] }; // ✅ 26-key model (a-z)
let stdoutBuffer = "";

// ============================================================
// 🔹 SSE (Server-Sent Events) STREAMING
// Tracks connected clients and pushes frames / events as they
// arrive from the Python process — no client polling required.
// ============================================================
const sseClients = new Set<Response>();

function pushSSE(event: Record<string, unknown>) {
  const payload = `data: ${JSON.stringify(event)}\n\n`;
  for (const client of sseClients) {
    try {
      client.write(payload);
    } catch {
      sseClients.delete(client);
    }
  }
}

const SAVE_DIR = path.join(process.cwd(), "ml/results_csv");
const EXPECTED_PATH = path.join(SAVE_DIR, "expected_words.json");

// ============================================================
// ✅ START DETECTION
// ============================================================

router.post("/start", (req: Request, res: Response) => {
  if (pythonProcess) {
    return res.status(400).json({ message: "Detection is already running." });
  }

  // ✅ FULL STATE RESET: Essential for recalibration support
  // This ensures clean slate when user clicks "Recalibrate"
  latestDetection = { key: "", finger: "", hand: "", correct: null };
  latestFrame = null;
  calibrationState = { done: false, detected: 0, required: 26, detected_keys: [] }; // ✅ 26-key model
  stdoutBuffer = "";

  const scriptPath = path.join(__dirname, "../../ml/notebooks/detect_keyboard_live.py");
  console.log("🚀 Starting Python detection (headless):", scriptPath);

  pythonProcess = spawn("python", ["-u", scriptPath], {
    cwd: path.dirname(scriptPath),
    windowsHide: true,
  });

  // ============================================================
  // FRAME STATISTICS — For performance monitoring WITHOUT logging raw data
  // ============================================================
  let frameCount = 0;
  let lastFrameLog = Date.now();
  let bytesProcessed = 0;

  // Line-buffered stdout parser — handles large frame payloads that may
  // arrive across multiple 'data' chunks
  pythonProcess.stdout.on("data", (data: Buffer) => {
    stdoutBuffer += data.toString();
    const lines = stdoutBuffer.split("\n");
    stdoutBuffer = lines.pop() || ""; // retain any incomplete trailing line

    for (const line of lines) {
      const text = line.trim();
      if (!text) continue;

      try {
        const parsed = JSON.parse(text);

        // ============================================================
        // CRITICAL: DO NOT LOG RAW FRAME DATA
        // Only log metadata for frames, full content for other events
        // ============================================================
        switch (parsed.type) {
          case "frame":
            // ✅ Track frame statistics WITHOUT logging Base64 data
            frameCount++;
            bytesProcessed += text.length;

            // Log frame stats every 5 seconds (not every frame)
            const now = Date.now();
            if (now - lastFrameLog >= 5000) {
              const elapsed = (now - lastFrameLog) / 1000;
              const fps = (frameCount / elapsed).toFixed(1);
              const mbps = ((bytesProcessed / elapsed) / 1024 / 1024).toFixed(2);
              console.log(`📊 Frame stats: ${fps} FPS, ${mbps} MB/s, ${frameCount} frames processed`);
              frameCount = 0;
              bytesProcessed = 0;
              lastFrameLog = now;
            }

            // Send frame to frontend with fingertip count (no logging)
            latestFrame = parsed.frame || null;
            if (latestFrame) {
              pushSSE({
                type: "frame",
                frame: latestFrame,
                fingertip_count: parsed.fingertip_count || 0  // ✅ Forward live finger count
              });
            }
            break;

          case "calibration_progress":
            // Log calibration progress (contains frame, but we only log metadata)
            console.log(
              `🔄 Calibration: ${parsed.detected}/${parsed.required} keys detected`
            );
            calibrationState = {
              done: false,
              detected: parsed.detected,
              required: parsed.required,
              detected_keys: parsed.detected_keys || [],
              frame: parsed.frame || null,
            };
            pushSSE({
              type: "calibration_progress",
              detected: parsed.detected,
              required: parsed.required,
              detected_keys: parsed.detected_keys || [],
              frame: parsed.frame || null,
            });
            break;

          case "calibration_done":
            console.log(`✅ Calibration complete: ${parsed.locked_keys} keys locked`);
            calibrationState = {
              done: true,
              detected: parsed.locked_keys,
              required: 26, // ✅ 26-key model
              detected_keys: calibrationState.detected_keys,
            };
            latestFrame = null;
            // ✅ FIX: Forward locked_keys to frontend for validation
            pushSSE({ type: "calibration_done", locked_keys: parsed.locked_keys });
            break;

          case "error":
            console.log(`❌ Python error: ${parsed.message} (${parsed.reason || 'unknown'})`);
            latestDetection = {
              error: true,
              message: parsed.message,
              reason: parsed.reason,
            };
            pushSSE({ type: "error", message: parsed.message, reason: parsed.reason });
            break;

          case "detection":
            console.log(
              `⌨️  Keystroke: ${parsed.key} (${parsed.finger}, ${parsed.hand}) → ${parsed.ml_label}`
            );
            latestDetection = parsed;
            pushSSE(parsed);
            break;

          case "fps_stats":
            // Python's internal FPS monitoring (sent to stderr, but handled here if on stdout)
            console.log(
              `🎯 Python FPS: Visual=${parsed.visual_fps}, Inference=${parsed.inference_fps}`
            );
            break;

          default:
            // Log unknown event types for debugging
            if (parsed.key) {
              latestDetection = parsed;
              pushSSE(parsed);
            } else {
              console.log(`📟 [PYTHON]: ${JSON.stringify(parsed)}`);
            }
            break;
        }
      } catch {
        // Non-JSON messages (model loading logs, etc.) — log them as-is
        console.log(`📟 [PYTHON]: ${text}`);
      }
    }
  });

  // Non-fatal C++ warning patterns emitted by TFLite / MediaPipe internals
  // before absl logging initialises — these cannot be suppressed from Python
  // and do not affect inference accuracy.
  const BENIGN_STDERR_PATTERNS = [
    /inference_feedback_manager/,
    /absl::InitializeLog/,
    /TensorFlow Lite XNNPACK delegate/,
    /Using NORM_RECT without IMAGE_DIMENSIONS/,
    /^W\d{4}\s/,  // absl W#### warning prefix
    /^I\d{4}\s/,  // absl I#### info prefix
  ];

  pythonProcess.stderr.on("data", (data: Buffer) => {
    const lines = data.toString().trim().split("\n");
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;

      // Try to parse as JSON (FPS stats are sent to stderr)
      try {
        const parsed = JSON.parse(trimmed);
        if (parsed.type === "fps_stats") {
          console.log(
            `🎯 Python Performance: Visual=${parsed.visual_fps} FPS, Inference=${parsed.inference_fps} FPS` +
            ` (targets: ${parsed.target_visual}/${parsed.target_inference})`
          );
          continue;
        }
      } catch {
        // Not JSON, handle as regular stderr
      }

      // Only log lines that don't match known benign C++ noise
      const isBenign = BENIGN_STDERR_PATTERNS.some((re) => re.test(trimmed));
      if (!isBenign) {
        console.error(`⚠️ [PYTHON ERROR]: ${trimmed}`);
      }
    }
  });

  pythonProcess.on("close", (code: number) => {
    console.log(`🧠 Python process exited with code ${code}`);
    if (!calibrationState.done && code !== 0) {
      latestDetection = {
        error: true,
        message: "Detection process exited unexpectedly.",
      };
    }
    pythonProcess = null;
  });

  res.json({ message: "Keyboard detection started." });
});

// ============================================================
// ✅ STOP DETECTION
// ============================================================

router.post("/stop", (req: Request, res: Response) => {
  if (!pythonProcess) {
    return res.status(400).json({ message: "No detection process running." });
  }

  console.log("🛑 Sending EXIT command to Python...");

  // Ask Python to shut down cleanly via stdin
  try {
    pythonProcess.stdin.write("EXIT\n");
  } catch { }

  // Give Python time to flush and exit
  setTimeout(() => {
    if (pythonProcess) {
      console.log("💀 Forcing Python process to close...");
      pythonProcess.kill("SIGKILL");  // Works on Windows ALWAYS
      pythonProcess = null;
    }
  }, 300);

  res.json({ message: "Keyboard detection stopped." });
});

// ============================================================
// ✅ SSE STREAM ENDPOINT
// Frontend connects once; frames and events are pushed at up to
// 30 FPS without any polling.  EventSource auto-reconnects on
// network blips.
// ============================================================

router.get("/stream", (req: Request, res: Response) => {
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });

  sseClients.add(res);

  // Flush current state so the client isn't blank while waiting
  if (latestFrame) {
    res.write(`data: ${JSON.stringify({ type: "frame", frame: latestFrame })}\n\n`);
  }
  if (!calibrationState.done) {
    res.write(
      `data: ${JSON.stringify({
        type: "calibration_progress",
        detected: calibrationState.detected,
        required: calibrationState.required,
        detected_keys: calibrationState.detected_keys,
        frame: calibrationState.frame || null,
      })}\n\n`
    );
  }

  req.on("close", () => {
    sseClients.delete(res);
  });
});

// ============================================================
// ✅ LIVE STATUS (Detection + Calibration + Visualization Frame)
// ============================================================

router.get("/status", (req: Request, res: Response) => {
  res.json({
    ...latestDetection,
    frame: latestFrame || calibrationState.frame || null,
    calibration: {
      done: calibrationState.done,
      detected: calibrationState.detected,
      required: calibrationState.required,
      detected_keys: calibrationState.detected_keys,
    },
  });
});

// ============================================================
// ✅ GET RESULTS (CSV list)
// ============================================================

router.get("/results", (req: Request, res: Response) => {
  if (!fs.existsSync(SAVE_DIR)) {
    return res.status(404).json({ message: "Results folder not found." });
  }

  const csvFiles = fs.readdirSync(SAVE_DIR).filter((f) => f.endsWith(".csv"));
  res.json({ files: csvFiles });
});

// ============================================================
// ✅ NEW: RECEIVE EXPECTED WORDS FROM FRONTEND
// ============================================================

router.post("/set-expected", (req: Request, res: Response) => {
  const { words } = req.body;
  if (!words || !Array.isArray(words)) {
    return res.status(400).json({ message: "Invalid data format" });
  }

  fs.mkdirSync(SAVE_DIR, { recursive: true });
  fs.writeFileSync(EXPECTED_PATH, JSON.stringify({ words }), "utf-8");
  console.log("✅ Expected typing words updated:", words.slice(0, 10), "...");
  res.json({ message: "Expected typing data saved" });
});

// ============================================================
// 🔹 LOAD TEXT DATA (Levels 1–4)
// ============================================================

const dataDir = path.join(__dirname, "../../data");

// Safe helper to load text file into array
function loadFile(filename: string, separator: string | RegExp = "\n") {
  try {
    const filePath = path.join(dataDir, filename);
    const content = fs.readFileSync(filePath, "utf-8");

    const chunks = content
      .split(/\r?\n\r?\n+/)
      .flatMap((block) => block.split(/\r?\n/))
      .map((line) => line.trim())
      .filter((line) => line.length > 0);

    return chunks;
  } catch (error) {
    console.error(`⚠️ Error loading ${filename}:`, error);
    return [];
  }
}

// ✅ Load data files
const letterList = loadFile("letters.txt");
const wordList = loadFile("words.txt");
const phraseList = loadFile("phrases.txt");
const sentenceList = loadFile("sentences.txt");

// ============================================================
// 🔹 Helper Function for Random Items
// ============================================================

const getRandomItems = (arr: string[], count: number) => {
  if (arr.length === 0) return [];
  const selected: string[] = [];
  for (let i = 0; i < count; i++) {
    const item = arr[Math.floor(Math.random() * arr.length)];
    selected.push(item);
  }
  return selected;
};

// ============================================================
// 🔹 TEXT LEVEL ROUTE (Letters → Sentences)
// ============================================================

router.get("/level/:id", (req: Request, res: Response) => {
  const { id } = req.params;
  let data: string[] = [];

  switch (id) {
    case "1":
      // 🔹 Level 1: Letters
      data = getRandomItems(letterList, 10);
      break;

    case "2":
      // 🔹 Level 2: Words
      const numWords = Math.floor(Math.random() * 3) + 3; // 3–5
      data = [getRandomItems(wordList, numWords).join(" ")];
      break;

    case "3":
      // 🔹 Level 3: Phrases
      const randomPhrase = phraseList[Math.floor(Math.random() * phraseList.length)];
      data = [randomPhrase];
      break;

    case "4":
      // 🔹 Level 4: Sentences
      const randomSentence = sentenceList[Math.floor(Math.random() * sentenceList.length)];
      data = [randomSentence];
      break;

    default:
      return res.status(404).json({ message: "Invalid level" });
  }

  res.json({ level: Number(id), data });
});

// ============================================================
// 🔹 EXPORT ROUTER
// ============================================================

export default router;
