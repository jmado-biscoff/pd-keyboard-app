//comment
import "dotenv/config";

export const MODEL_PATH = process.env.MODEL_PATH!;
export const FEATURE_CFG = process.env.FEATURE_CFG!;
export const PORT = process.env.PORT || 5000;
export const MONGO_URI = process.env.MONGO_URI!;
export const JWT_SECRET = process.env.JWT_SECRET!;

if (!MODEL_PATH || !FEATURE_CFG) {
  throw new Error("MODEL_PATH and FEATURE_CFG must be set in .env");
}

if (!MONGO_URI) {
  throw new Error("MONGO_URI must be set in .env");
}

if (!JWT_SECRET) {
  throw new Error("JWT_SECRET must be set in .env");
}

console.log("[ENV] MODEL_PATH =", MODEL_PATH);
console.log("[ENV] FEATURE_CFG =", FEATURE_CFG);
console.log("[ENV] MONGO_URI =", MONGO_URI);
console.log("[ENV] JWT_SECRET loaded ✓");
