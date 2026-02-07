import express, { Request, Response } from "express";
import { spawn } from "child_process";
import path from "path";
import fs from "fs";

const router = express.Router();

// State management
let pythonProcess: any = null;
let latestDetection: any = { key: "", finger: "", hand: "", correct: null };
let latestFrame: string | null = null;
let calibrationState: {
  done: boolean;
  detected: number;
  required: number;
  detected_keys: string[];
  frame?: string | null;
} = { done: false, detected: 0, required: 26, detected_keys: [] };
let stdoutBuffer = "";

// SSE client connections
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

// Start detection
router.post("/start", (req: Request, res: Response) => {
  if (pythonProcess) {
    return res.status(400).json({ message: "Detection is already running." });
  }

  // Full state reset for recalibration support
  latestDetection = { key: "", finger: "", hand: "", correct: null };
  latestFrame = null;
  calibrationState = { done: false, detected: 0, required: 26, detected_keys: [] };
  stdoutBuffer = "";

  const scriptPath = path.join(__dirname, "../../ml/notebooks/detect_keyboard_live.py");
  console.log("Starting Python detection (headless):", scriptPath);

  pythonProcess = spawn("python", ["-u", scriptPath], {
    cwd: path.dirname(scriptPath),
    windowsHide: true,
  });

  // Frame statistics for performance monitoring
  let frameCount = 0;
  let lastFrameLog = Date.now();
  let bytesProcessed = 0;

  // Line-buffered stdout parser
  pythonProcess.stdout.on("data", (data: Buffer) => {
    stdoutBuffer += data.toString();
    const lines = stdoutBuffer.split("\n");
    stdoutBuffer = lines.pop() || "";

    for (const line of lines) {
      const text = line.trim();
      if (!text) continue;

      try {
        const parsed = JSON.parse(text);

        switch (parsed.type) {
          case "frame":
            frameCount++;
            bytesProcessed += text.length;

            const now = Date.now();
            if (now - lastFrameLog >= 5000) {
              const elapsed = (now - lastFrameLog) / 1000;
              const fps = (frameCount / elapsed).toFixed(1);
              const mbps = ((bytesProcessed / elapsed) / 1024 / 1024).toFixed(2);
              console.log(`Frame stats: ${fps} FPS, ${mbps} MB/s, ${frameCount} frames processed`);
              frameCount = 0;
              bytesProcessed = 0;
              lastFrameLog = now;
            }

            latestFrame = parsed.frame || null;
            if (latestFrame) {
              pushSSE({
                type: "frame",
                frame: latestFrame,
                fingertip_count: parsed.fingertip_count || 0,
                left_fingers_count: parsed.left_fingers_count || 0,
                right_fingers_count: parsed.right_fingers_count || 0
              });
            }
            break;

          case "calibration_progress":
            console.log(`Calibration: ${parsed.detected}/${parsed.required} keys detected`);
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
            console.log(`Calibration complete: ${parsed.locked_keys} keys locked`);
            calibrationState = {
              done: true,
              detected: parsed.locked_keys,
              required: 26,
              detected_keys: calibrationState.detected_keys,
            };
            latestFrame = null;
            pushSSE({ type: "calibration_done", locked_keys: parsed.locked_keys });
            break;

          case "error":
            console.log(`Python error: ${parsed.message} (${parsed.reason || 'unknown'})`);
            latestDetection = {
              error: true,
              message: parsed.message,
              reason: parsed.reason,
            };
            pushSSE({ type: "error", message: parsed.message, reason: parsed.reason });
            break;

          case "detection":
            console.log(`Keystroke: ${parsed.key} (${parsed.finger}, ${parsed.hand}) → ${parsed.ml_label}`);
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
              console.log(`[PYTHON]: ${JSON.stringify(parsed)}`);
            }
            break;
        }
      } catch {
        console.log(`[PYTHON]: ${text}`);
      }
    }
  });

  // Benign C++ warnings from TFLite/MediaPipe internals
  const BENIGN_STDERR_PATTERNS = [
    /inference_feedback_manager/,
    /absl::InitializeLog/,
    /TensorFlow Lite XNNPACK delegate/,
    /Using NORM_RECT without IMAGE_DIMENSIONS/,
    /^W\d{4}\s/,
    /^I\d{4}\s/,
  ];

  pythonProcess.stderr.on("data", (data: Buffer) => {
    const lines = data.toString().trim().split("\n");
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;

      try {
        const parsed = JSON.parse(trimmed);
        if (parsed.type === "fps_stats") {
          console.log(`Python Performance: Visual=${parsed.visual_fps} FPS, Inference=${parsed.inference_fps} FPS (targets: ${parsed.target_visual}/${parsed.target_inference})`);
          continue;
        }
      } catch {
        // Not JSON
      }

      const isBenign = BENIGN_STDERR_PATTERNS.some((re) => re.test(trimmed));
      if (!isBenign) {
        console.error(`[PYTHON ERROR]: ${trimmed}`);
      }
    }
  });

  pythonProcess.on("close", (code: number) => {
    console.log(`Python process exited with code ${code}`);
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

// Stop detection
router.post("/stop", (req: Request, res: Response) => {
  if (!pythonProcess) {
    return res.status(400).json({ message: "No detection process running." });
  }

  console.log("Sending EXIT command to Python...");

  try {
    pythonProcess.stdin.write("EXIT\n");
  } catch { }

  setTimeout(() => {
    if (pythonProcess) {
      console.log("Forcing Python process to close...");
      pythonProcess.kill("SIGKILL");
      pythonProcess = null;
    }
  }, 300);

  res.json({ message: "Keyboard detection stopped." });
});

// SSE stream endpoint


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

// Live status
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

// Get results
router.get("/results", (req: Request, res: Response) => {
  if (!fs.existsSync(SAVE_DIR)) {
    return res.status(404).json({ message: "Results folder not found." });
  }
  const csvFiles = fs.readdirSync(SAVE_DIR).filter((f) => f.endsWith(".csv"));
  res.json({ files: csvFiles });
});

// Set expected words
router.post("/set-expected", (req: Request, res: Response) => {
  const { words, startIndex } = req.body;
  if (!words || !Array.isArray(words)) {
    return res.status(400).json({ message: "Invalid data format" });
  }
  const index = typeof startIndex === 'number' ? startIndex : 0;
  fs.mkdirSync(SAVE_DIR, { recursive: true });
  fs.writeFileSync(EXPECTED_PATH, JSON.stringify({ words, startIndex: index }), "utf-8");
  console.log(`Expected typing words updated: ${words.slice(0, 10).join(",")}... (starting at index ${index})`);
  res.json({ message: "Expected typing data saved" });
});

// ============================================================
// NOTE: Level generation logic has been moved to typingRoutes.ts
// This route file focuses solely on detection-related endpoints
// ============================================================

export default router;
