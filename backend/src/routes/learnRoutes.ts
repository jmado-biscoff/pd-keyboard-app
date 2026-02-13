import express, { Request, Response } from "express";
import fs from "fs";
import path from "path";

const router = express.Router();
const dataDir = path.join(__dirname, "../../data");

// Safe helper to load text file into array (same pattern as typingRoutes.ts)
function loadFile(filename: string): string[] {
  try {
    const filePath = path.join(dataDir, filename);
    const content = fs.readFileSync(filePath, "utf-8");
    return content
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line.length > 0);
  } catch (error) {
    console.error(`Error loading ${filename}:`, error);
    return [];
  }
}

// Load data files
const module1Drills = loadFile("module1_homerow.txt");
const module2Drills = loadFile("module2_toprow.txt");
const module3Drills = loadFile("module3_bottomrow.txt");
const allLetters = loadFile("letters.txt");
const wordList = loadFile("words.txt");

// Pick N random items from array (with shuffling, no repeats)
function pickRandom(arr: string[], count: number): string[] {
  const shuffled = [...arr].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, Math.min(count, shuffled.length));
}

// Generate a random letter sequence of given length
function generateRandomLetters(letters: string[], length: number): string {
  let result = "";
  for (let i = 0; i < length; i++) {
    result += letters[Math.floor(Math.random() * letters.length)];
  }
  return result;
}

// Module metadata
const MODULE_TITLES: Record<number, string> = {
  1: "Home Row Heroes",
  2: "Top Row Adventure",
  3: "Bottom Row Explorer",
  4: "Alphabet Mastery",
  5: "Word Builder",
};

// ============================================================
// GET /api/learn/module/:id
// Returns { moduleId, title, drills: string[] }
// ============================================================
router.get("/module/:id", (req: Request, res: Response) => {
  const id = parseInt(req.params.id);
  let drills: string[] = [];

  switch (id) {
    case 1:
      drills = pickRandom(module1Drills, 10);
      break;
    case 2:
      drills = pickRandom(module2Drills, 10);
      break;
    case 3:
      drills = pickRandom(module3Drills, 10);
      break;
    case 4:
      // Generate a single 30-character random sequence from all letters
      drills = [generateRandomLetters(allLetters, 30)];
      break;
    case 5:
      // Pick 10 random words (auto-advance, no spaces needed)
      drills = pickRandom(wordList, 10);
      break;
    default:
      return res.status(404).json({ message: "Invalid module ID" });
  }

  res.json({
    moduleId: id,
    title: MODULE_TITLES[id] || `Module ${id}`,
    drills,
  });
});

export default router;
