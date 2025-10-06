import "dotenv/config";

export const MODEL_PATH = process.env.MODEL_PATH!;
export const FEATURE_CFG = process.env.FEATURE_CFG!;

if (!MODEL_PATH || !FEATURE_CFG) {
  throw new Error("MODEL_PATH and FEATURE_CFG must be set in .env");
}

console.log("[ENV] MODEL_PATH =", MODEL_PATH);
console.log("[ENV] FEATURE_CFG =", FEATURE_CFG);
