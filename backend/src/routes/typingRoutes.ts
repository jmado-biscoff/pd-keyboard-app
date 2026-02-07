import express, { Request, Response } from "express";
import fs from "fs";
import path from "path";

const router = express.Router();

// ============================================================
// 🔹 Load Text Data (Levels 1–2)
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

// ✅ Load data files
const letterList = loadFile("letters.txt");
const wordList = loadFile("words.txt");

// ============================================================
// 🔹 Helper Functions
// ============================================================

// Generate 50 random characters (can repeat consecutively)
const generate50RandomChars = (arr: string[]) => {
  if (arr.length === 0) return "";
  let result = "";
  for (let i = 0; i < 50; i++) {
    const char = arr[Math.floor(Math.random() * arr.length)];
    result += char;
  }
  return result;
};

// Generate random words totaling exactly 50 characters (no word repeats)
const generate50CharWords = (arr: string[]) => {
  if (arr.length === 0) return "";

  const availableWords = [...arr]; // Copy to track unused words
  const selectedWords: string[] = [];
  let currentLength = 0;

  // Keep trying to build a 50-character string
  while (currentLength < 50 && availableWords.length > 0) {
    // Calculate remaining space
    const remaining = 50 - currentLength;

    // If we have at least one word selected, account for the space separator
    const spaceForNextWord = selectedWords.length > 0 ? remaining - 1 : remaining;

    // Find words that would fit in the remaining space
    const fittingWords = availableWords.filter(word => word.length <= spaceForNextWord);

    if (fittingWords.length === 0) {
      // No words fit perfectly, restart the algorithm
      selectedWords.length = 0;
      currentLength = 0;
      availableWords.length = 0;
      availableWords.push(...arr);
      continue;
    }

    // Pick a random word from fitting options
    const randomIndex = Math.floor(Math.random() * fittingWords.length);
    const selectedWord = fittingWords[randomIndex];

    // Add word to result
    selectedWords.push(selectedWord);
    currentLength += selectedWord.length + (selectedWords.length > 1 ? 1 : 0); // +1 for space

    // Remove selected word from available pool
    const wordIndex = availableWords.indexOf(selectedWord);
    availableWords.splice(wordIndex, 1);
  }

  // Join with spaces and verify length
  const result = selectedWords.join(" ");

  // If we hit exactly 50 characters, return it
  if (result.length === 50) {
    return result;
  }

  // Otherwise, try again recursively (rare case)
  return generate50CharWords(arr);
};

// ============================================================
// 🔹 API ROUTE
// ============================================================

router.get("/level/:id", (req: Request, res: Response) => {
  const { id } = req.params;
  let data: string[] = [];

  switch (id) {
    case "1":
      // 🔹 Level 1: 50 random characters (can repeat consecutively)
      const randomChars = generate50RandomChars(letterList);
      data = [randomChars];
      break;

    case "2":
      // 🔹 Level 2: Random words totaling exactly 50 characters (no word repeats)
      const randomWords = generate50CharWords(wordList);
      data = [randomWords];
      break;

    default:
      return res.status(404).json({ message: "Invalid level" });
  }

  res.json({ level: Number(id), data });
});

// ============================================================
// 🔹 Export
// ============================================================

export default router;
