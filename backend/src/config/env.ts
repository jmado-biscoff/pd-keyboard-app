import "dotenv/config";

export const PORT = process.env.PORT || 5000;
export const MONGO_URI = process.env.MONGO_URI!;
export const JWT_SECRET = process.env.JWT_SECRET!;

if (!MONGO_URI) {
  throw new Error("MONGO_URI must be set in .env");
}

if (!JWT_SECRET) {
  throw new Error("JWT_SECRET must be set in .env");
}
