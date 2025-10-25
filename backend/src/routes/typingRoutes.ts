import express, { Request, Response } from "express";
import fs from "fs";
import path from "path";

const router = express.Router();

// ============================================================
// 🔹 Load Text Data (Levels 1–4)
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
const phraseList = loadFile("phrases.txt");
const sentenceList = loadFile("sentences.txt");

// ============================================================
// 🔹 Helper Function
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
// 🔹 API ROUTE
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
      // 🔹 Level 3: Phrases (single random phrase)
      const randomPhrase =
        phraseList[Math.floor(Math.random() * phraseList.length)];
      data = [randomPhrase];
      break;

    case "4":
      // 🔹 Level 4: Sentences
      const randomSentence =
        sentenceList[Math.floor(Math.random() * sentenceList.length)];
      data = [randomSentence];
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