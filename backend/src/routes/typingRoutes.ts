import express, { Request, Response } from "express";
import fs from "fs";
import path from "path";

const router = express.Router();

// ============================================================
// Load Text Data (Levels 1–2)
// ============================================================

const dataDir = path.join(__dirname, "../../data");

// Safe helper to load text file into array
function loadFile(filename: string, separator: string | RegExp = "\n") {
  try {
    const filePath = path.join(dataDir, filename);
    const content = fs.readFileSync(filePath, "utf-8");

    // Smart split: handles both single and double newlines cleanly
    const chunks = content
      .split(/\r?\n\r?\n+/) // split by double newlines first
      .flatMap((block) => block.split(/\r?\n/)) // fallback for single newlines
      .map((line) => line.trim())
      .filter((line) => line.length > 0);

    return chunks;
  } catch (error) {
    console.error(`⚠️ Error loading ${filename}:`, error);
    return [];
  }
}

// Load data files
const letterList = loadFile("letters.txt");
const wordList = loadFile("words.txt");

// ============================================================
// Helper Functions
// ============================================================

// Generate 50 random letters (can repeat consecutively)
const generate50RandomChars = (arr: string[]) => {
  if (arr.length === 0) return "";
  let result = "";
  for (let i = 0; i < 50; i++) {
    const char = arr[Math.floor(Math.random() * arr.length)];
    result += char;
  }
  return result;
};

// Generate random words containing exactly 50 letters (excluding spaces)
const generate50LetterWords = (arr: string[]): string => {
  if (arr.length === 0) return "";

  const availableWords = [...arr];
  const selectedWords: string[] = [];
  let letterCount = 0;

  while (letterCount < 50 && availableWords.length > 0) {
    const remaining = 50 - letterCount;

    // Find words that fit within the remaining letter count
    const fittingWords = availableWords.filter(word => word.length <= remaining);

    if (fittingWords.length === 0) {
      // No words fit — restart
      selectedWords.length = 0;
      letterCount = 0;
      availableWords.length = 0;
      availableWords.push(...arr);
      continue;
    }

    // Pick a random word from fitting options
    const randomIndex = Math.floor(Math.random() * fittingWords.length);
    const selectedWord = fittingWords[randomIndex];

    selectedWords.push(selectedWord);
    letterCount += selectedWord.length;

    // Remove selected word from available pool (no repeats)
    const wordIndex = availableWords.indexOf(selectedWord);
    availableWords.splice(wordIndex, 1);
  }

  // Verify exactly 50 letters
  if (letterCount === 50) {
    return selectedWords.join(" ");
  }

  // Retry if not exact (rare case)
  return generate50LetterWords(arr);
};

// ============================================================
// API ROUTE
// ============================================================

router.get("/level/:id", (req: Request, res: Response) => {
  const { id } = req.params;
  let data: string[] = [];

  switch (id) {
    case "1":
      // Level 1: 50 random letters (can repeat consecutively)
      const randomChars = generate50RandomChars(letterList);
      data = [randomChars];
      break;

    case "2":
      // Level 2: Random words containing exactly 50 letters (spaces are extra)
      const randomWords = generate50LetterWords(wordList);
      data = [randomWords];
      break;

    default:
      return res.status(404).json({ message: "Invalid level" });
  }

  res.json({ level: Number(id), data });
});

// ============================================================
// Export
// ============================================================

export default router;
