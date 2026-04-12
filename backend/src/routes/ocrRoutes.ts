import { Router, Request, Response } from "express";
import { ImageAnnotatorClient } from "@google-cloud/vision";
import path from "path";
import axios from "axios";
import fs from "fs";

const router = Router();

// Determine authentication method and create appropriate client
type AuthMethod = "service-account-json" | "service-account-file" | "api-key" | "none";

function getAuthMethod(): { method: AuthMethod; config: any } {
  // Priority 1: Service account JSON from environment variable
  if (process.env.GOOGLE_CREDENTIALS_JSON) {
    try {
      const credentials = JSON.parse(process.env.GOOGLE_CREDENTIALS_JSON);
      console.log("✅ OCR Auth: Using service account from GOOGLE_CREDENTIALS_JSON");
      return { method: "service-account-json", config: { credentials } };
    } catch (err) {
      console.error("❌ Failed to parse GOOGLE_CREDENTIALS_JSON:", err);
    }
  }

  // Priority 2: API key from environment variable
  if (process.env.GOOGLE_VISION_API_KEY) {
    console.log("✅ OCR Auth: Using API key from GOOGLE_VISION_API_KEY");
    return { method: "api-key", config: { apiKey: process.env.GOOGLE_VISION_API_KEY } };
  }

  // Priority 3: Local service account file (development)
  const keyFilePath = path.join(process.cwd(), "expomapapp-475414-c637430cbe92.json");
  try {
    if (fs.existsSync(keyFilePath)) {
      console.log("✅ OCR Auth: Using local service account file:", keyFilePath);
      return { method: "service-account-file", config: { keyFilename: keyFilePath } };
    }
  } catch (err) {
    // Ignore file check errors
  }

  console.warn("⚠️  OCR Auth: No Google Vision credentials found");
  return { method: "none", config: {} };
}

const { method: authMethod, config: clientConfig } = getAuthMethod();

// Create client for service account authentication
let visionClient: ImageAnnotatorClient | null = null;
if (authMethod === "service-account-json" || authMethod === "service-account-file") {
  visionClient = new ImageAnnotatorClient(clientConfig);
}

/**
 * Call Google Vision API using REST (for API key auth)
 */
async function detectTextWithApiKey(base64Image: string, apiKey: string): Promise<string> {
  const url = `https://vision.googleapis.com/v1/images:annotate?key=${apiKey}`;

  const body = {
    requests: [
      {
        image: {
          content: base64Image,
        },
        features: [
          {
            type: "TEXT_DETECTION",
          },
        ],
      },
    ],
  };

  try {
    const response = await axios.post<{ responses: Array<{ fullTextAnnotation?: { text: string } }> }>(url, body, {
      headers: {
        "Content-Type": "application/json",
      },
    });

    const textAnnotation = response.data?.responses?.[0]?.fullTextAnnotation?.text || "";
    return textAnnotation;
  } catch (error: any) {
    if (error.response) {
      const errorMessage = error.response.data?.error?.message || error.message;
      throw new Error(`Google Vision API error (${error.response.status}): ${errorMessage}`);
    }
    throw error;
  }
}

/**
 * Call Google Vision API using client library (for service account auth)
 */
async function detectTextWithClient(base64Image: string): Promise<string> {
  if (!visionClient) {
    throw new Error("Vision client not initialized");
  }

  const [result] = await visionClient.textDetection({
    image: { content: base64Image },
  });

  return result.fullTextAnnotation?.text || "";
}

router.post("/parse", async (req: Request, res: Response) => {
  try {
    const { base64Image } = req.body;

    // Validate input
    if (!base64Image) {
      return res.status(400).json({
        error: "base64Image is required",
        ParsedResults: [{ ParsedText: "", ErrorMessage: "No image provided" }]
      });
    }

    if (typeof base64Image !== "string") {
      return res.status(400).json({
        error: "base64Image must be a string",
        ParsedResults: [{ ParsedText: "", ErrorMessage: "Invalid image format" }]
      });
    }

    // Clean base64 data (remove data URL prefix if present)
    const cleanBase64 = base64Image.replace(/^data:image\/[a-z]+;base64,/i, "");

    // Basic base64 validation (allow for standard base64 characters)
    if (!/^[A-Za-z0-9+/]*={0,2}$/.test(cleanBase64.replace(/\s/g, ""))) {
      return res.status(400).json({
        error: "Invalid base64 format",
        ParsedResults: [{ ParsedText: "", ErrorMessage: "Invalid base64 encoding" }]
      });
    }

    console.log(`🔍 Processing OCR request (auth method: ${authMethod})...`);

    let parsedText: string;

    // Use appropriate method based on auth type
    if (authMethod === "api-key") {
      // Use REST API for API key authentication
      parsedText = await detectTextWithApiKey(cleanBase64, clientConfig.apiKey);
    } else if (authMethod === "service-account-json" || authMethod === "service-account-file") {
      // Use client library for service account authentication
      parsedText = await detectTextWithClient(cleanBase64);
    } else {
      return res.status(500).json({
        error: "Google Vision API not configured",
        details: "Set GOOGLE_CREDENTIALS_JSON or GOOGLE_VISION_API_KEY environment variable",
        ParsedResults: [{ ParsedText: "", ErrorMessage: "OCR service not configured" }]
      });
    }

    if (!parsedText || !parsedText.trim()) {
      console.warn("⚠️  No text detected in image");
      return res.status(200).json({
        ParsedResults: [{
          ParsedText: "",
          ErrorMessage: "No text detected in image"
        }]
      });
    }

    console.log(`✅ OCR successful: extracted ${parsedText.length} characters`);

    // Return response in expected format
    res.json({
      ParsedResults: [{
        ParsedText: parsedText.trim(),
      }],
    });

  } catch (error: any) {
    console.error("❌ OCR Error:", error.message);

    // Handle specific error types
    if (error.message?.includes("403") || error.message?.includes("PERMISSION_DENIED")) {
      return res.status(403).json({
        error: "Google Vision API access denied. Check credentials and API permissions.",
        ParsedResults: [{ ParsedText: "", ErrorMessage: "Authentication failed" }]
      });
    }

    if (error.message?.includes("429") || error.message?.includes("RESOURCE_EXHAUSTED")) {
      return res.status(429).json({
        error: "Google Vision API quota exceeded. Please try again later.",
        ParsedResults: [{ ParsedText: "", ErrorMessage: "Quota exceeded" }]
      });
    }

    if (error.message?.includes("INVALID_ARGUMENT") || error.message?.includes("invalid")) {
      return res.status(400).json({
        error: "Invalid image data or format",
        ParsedResults: [{ ParsedText: "", ErrorMessage: "Invalid image" }]
      });
    }

    // Generic error
    res.status(500).json({
      error: "OCR processing failed",
      details: process.env.NODE_ENV === "development" ? error.message : undefined,
      ParsedResults: [{ ParsedText: "", ErrorMessage: "Processing failed" }]
    });
  }
});

export default router;