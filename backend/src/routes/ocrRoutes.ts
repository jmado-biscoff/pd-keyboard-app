import { Router, Request, Response } from "express";
import { ImageAnnotatorClient } from "@google-cloud/vision";
import path from "path";

const router = Router();

// Initialize the Google Vision client
// In production (Railway), we use the GOOGLE_CREDENTIALS_JSON env var string
// In development, we fallback to the local service account file
const clientConfig: any = {};

if (process.env.GOOGLE_CREDENTIALS_JSON) {
  try {
    clientConfig.credentials = JSON.parse(process.env.GOOGLE_CREDENTIALS_JSON);
  } catch (err) {
    console.error("Error parsing GOOGLE_CREDENTIALS_JSON env var:", err);
  }
} else {
  clientConfig.keyFilename = path.join(process.cwd(), "expomapapp-475414-c637430cbe92.json");
}

const client = new ImageAnnotatorClient(clientConfig);

router.post("/parse", async (req: Request, res: Response) => {
  try {
    const { base64Image } = req.body;

    if (!base64Image) {
      return res.status(400).json({ error: "Base64 image is required" });
    }

    // Google Vision expects clean base64 data (without the data:image/jpeg;base64, prefix)
    const cleanBase64 = base64Image.replace(/^data:image\/[a-z]+;base64,/, "");

    // Prepare the request for the official library
    const [result] = await client.textDetection({
      image: { content: cleanBase64 },
    });

    const parsedText = result.fullTextAnnotation?.text || "";

    // Map the response format to match what the frontend expects
    const formattedData = {
      ParsedResults: [
        {
          ParsedText: parsedText,
        },
      ],
    };

    res.json(formattedData);
  } catch (error: any) {
    console.error("Google Vision Client Error:", error.message);
    res.status(500).json({ error: "Failed to process Google Vision OCR request" });
  }
});

export default router;
