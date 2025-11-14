// backend/src/__tests__/api.routes.test.ts
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import express from "express";
import request from "supertest";

// Put a mem store on globalThis so the vi.mock factory (which is hoisted)
// can access it without referencing not-yet-initialized variables.
const MEM_KEY = "__resultsMem__";
(globalThis as any)[MEM_KEY] = { results: [] as any[] };

// ──────────────────────────────────────────────────────────────
// Mock Result model (no external vars referenced in factory!)
// Chainable: Result.find().sort({date:-1}).limit(10)
// ──────────────────────────────────────────────────────────────
vi.mock("../models/Result", () => {
  class Result {
    _id: string;
    userId?: string;
    level: number;
    wpm: number;
    accuracy: number;
    grade?: string;
    sessionType?: string;
    date: Date;

    constructor(doc: Partial<Result>) {
      this._id = `id_${Math.random().toString(36).slice(2, 8)}`;
      this.userId = doc.userId;
      this.level = doc.level!;
      this.wpm = doc.wpm!;
      this.accuracy = doc.accuracy!;
      this.grade = doc.grade;
      this.sessionType = doc.sessionType;
      this.date = new Date();
    }

    async save() {
      (globalThis as any)[MEM_KEY].results.push(this);
    }

    static find() {
      return {
        sort(_by: any) {
          return {
            limit(n: number) {
              const list = (globalThis as any)[MEM_KEY].results as any[];
              const sorted = [...list].sort(
                (a, b) => b.date.getTime() - a.date.getTime()
              );
              return sorted.slice(0, n);
            },
          };
        },
      };
    }
  }

  // expose for tests (optional)
  (Result as any).__getMem = () => (globalThis as any)[MEM_KEY];

  return { default: Result };
});

// Import the real route you showed
import resultsRoutes from "../routes/resultsRoutes";

// Build an express app mounting only /api/results
const buildApp = () => {
  const app = express();
  app.use(express.json());
  app.use("/api/results", resultsRoutes);
  return app;
};

beforeEach(() => {
  (globalThis as any)[MEM_KEY].results.length = 0;
});
afterEach(() => {
  vi.restoreAllMocks();
});

// ──────────────────────────────────────────────────────────────
// Testcase 1: simulate saving typing test results
// ──────────────────────────────────────────────────────────────
describe("resultsRoutes - POST /api/results", () => {
  it("400 on missing fields; 201 on valid payload and persists", async () => {
    const app = buildApp();

    // 400 path
    const bad = await request(app).post("/api/results").send({ wpm: 30 });
    expect(bad.status).toBe(400);
    expect(bad.body.message).toMatch(/missing required fields/i);

    // 201 path
    const payload = {
      userId: "stu1",
      level: 1,
      wpm: 42,
      accuracy: 96,
      grade: "A",
      sessionType: "evaluated",
    };
    const ok = await request(app).post("/api/results").send(payload);
    expect(ok.status).toBe(201);
    expect(ok.body.level).toBe(1);
    expect(ok.body.wpm).toBe(42);

    const mem = (globalThis as any)[MEM_KEY].results as any[];
    expect(mem.length).toBe(1);
  });
});

// ──────────────────────────────────────────────────────────────
// Testcase 2: retrieve latest 10 results (sorted desc by date)
// ──────────────────────────────────────────────────────────────
describe("resultsRoutes - GET /api/results", () => {
  it("returns latest 10 results sorted by date desc", async () => {
    const app = buildApp();
    const mem = (globalThis as any)[MEM_KEY].results as any[];

    const now = Date.now();
    for (let i = 0; i < 12; i++) {
      mem.push({
        _id: `id_${i}`,
        userId: `u${i}`,
        level: 1,
        wpm: 20 + i, // 20..31
        accuracy: 90 + (i % 5),
        grade: i % 2 ? "B" : "A",
        sessionType: "practice",
        date: new Date(now - (12 - i) * 1000), // older → newer
        save: async () => {},
      });
    }

    const res = await request(app).get("/api/results");
    expect(res.status).toBe(200);
    expect(Array.isArray(res.body)).toBe(true);
    expect(res.body.length).toBe(10);

    const wpms = res.body.map((r: any) => r.wpm);
    expect(wpms[0]).toBe(31); // newest first
    expect(wpms[wpms.length - 1]).toBe(22);
  });
});

// ──────────────────────────────────────────────────────────────
// Testcase 3: simulated server error on POST (save throws)
// ──────────────────────────────────────────────────────────────
describe("resultsRoutes - POST /api/results (error path)", () => {
  it("500 when save() rejects", async () => {
    const app = buildApp();
    const Result = (await import("../models/Result")).default as any;

    const saveSpy = vi
      .spyOn(Result.prototype, "save")
      .mockRejectedValueOnce(new Error("DB down"));

    const payload = {
      userId: "stuX",
      level: 2,
      wpm: 55,
      accuracy: 98,
      grade: "A",
      sessionType: "evaluated",
    };

    const res = await request(app).post("/api/results").send(payload);
    expect(res.status).toBe(500);
    expect(res.body.message).toMatch(/failed to save result/i);

    saveSpy.mockRestore();
  });
});
