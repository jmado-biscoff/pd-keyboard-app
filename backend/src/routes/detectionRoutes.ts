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

const SAVE_DIR = path.join(process.cwd(), "ml/results_csv");
const EXPECTED_PATH = path.join(SAVE_DIR, "expected_words.json");

// ============================================================
// ✅ START DETECTION
// ============================================================

router.post("/start", (req: Request, res: Response) => {
  if (pythonProcess) {
    return res.status(400).json({ message: "Detection is already running." });
  }

  const scriptPath = path.join(__dirname, "../../ml/notebooks/detect_keyboard_live.py");
  console.log("🚀 Starting Python detection:", scriptPath);

  pythonProcess = spawn("python", [scriptPath], {
    cwd: path.dirname(scriptPath),
    shell: true,
  });

  pythonProcess.stdout.on("data", (data: Buffer) => {
    const text = data.toString().trim();
    console.log(`📟 [PYTHON]: ${text}`);

    try {
      const parsed = JSON.parse(text);
      if (parsed.key) latestDetection = parsed;
    } catch {
      // ignore non-JSON messages
    }
  });

  pythonProcess.stderr.on("data", (data: Buffer) => {
    console.error(`⚠️ [PYTHON ERROR]: ${data.toString().trim()}`);
  });

  pythonProcess.on("close", (code: number) => {
    console.log(`🧠 Python process exited with code ${code}`);
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

  pythonProcess.kill("SIGINT");
  pythonProcess = null;
  console.log("🛑 Python detection stopped.");
  res.json({ message: "Keyboard detection stopped." });
});

// ============================================================
// ✅ LIVE STATUS (Finger Correctness)
// ============================================================

router.get("/status", (req: Request, res: Response) => {
  res.json(latestDetection);
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
