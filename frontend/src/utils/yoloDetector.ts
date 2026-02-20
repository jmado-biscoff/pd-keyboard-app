// Client-side YOLOv8 ONNX inference for keyboard key detection
// Replaces server-side Python YOLO calibration with browser-only detection

// Lazy-loaded to avoid blocking module initialization
let ort: typeof import("onnxruntime-web") | null = null;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let session: any = null;

// YOLO configuration
const MODEL_URL = "/models/yolo_keyboard.onnx";
const INPUT_SIZE = 640;
const CONF_THRESHOLD = 0.4;
const IOU_THRESHOLD = 0.45;

// Label fix from the Python pipeline (y and z are swapped in training data)
const YOLO_LABEL_FIX: Record<string, string> = { y: "z", z: "y" };

// 60 class names from the ONNX model metadata
const CLASS_NAMES: Record<number, string> = {
  0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
  10: "a", 11: "accent", 12: "ae", 13: "alt-left", 14: "altgr-right",
  15: "b", 16: "c", 17: "caret", 18: "comma", 19: "d", 20: "del",
  21: "e", 22: "enter", 23: "f", 24: "g", 25: "h", 26: "hash",
  27: "i", 28: "j", 29: "k", 30: "keyboard", 31: "l", 32: "less",
  33: "m", 34: "minus", 35: "n", 36: "o", 37: "oe", 38: "p", 39: "plus",
  40: "point", 41: "q", 42: "r", 43: "s", 44: "shift-left", 45: "shift-lock",
  46: "shift-right", 47: "space", 48: "ss", 49: "strg-left", 50: "strg-right",
  51: "t", 52: "tab", 53: "u", 54: "ue", 55: "v", 56: "w", 57: "x", 58: "y", 59: "z",
};

const NUM_CLASSES = 60;

export interface YoloDetection {
  label: string;       // corrected key label (a-z)
  confidence: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/**
 * Initialize the YOLO model for client-side key detection.
 * Downloads the ONNX model (~44 MB) on first call.
 */
export async function initYolo(): Promise<void> {
  if (session) return;

  if (!ort) {
    ort = await import("onnxruntime-web");
  }

  // Use WebGL backend for GPU acceleration (WASM fallback is automatic)
  session = await ort.InferenceSession.create(MODEL_URL, {
    executionProviders: ["webgl", "wasm"],
  });
}

/**
 * Only return true for single alphabetic characters (a-z keys).
 */
function isAllowedKey(label: string): boolean {
  return label.length === 1 && label >= "a" && label <= "z";
}

/**
 * Compute Intersection over Union for two boxes [x1, y1, x2, y2].
 */
function iou(a: number[], b: number[]): number {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (areaA + areaB - inter + 1e-6);
}

/**
 * Non-Maximum Suppression to remove overlapping detections.
 */
function nms(detections: YoloDetection[]): YoloDetection[] {
  // Sort by confidence descending
  const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
  const kept: YoloDetection[] = [];

  for (const det of sorted) {
    let dominated = false;
    for (const k of kept) {
      if (iou([det.x1, det.y1, det.x2, det.y2], [k.x1, k.y1, k.x2, k.y2]) > IOU_THRESHOLD) {
        dominated = true;
        break;
      }
    }
    if (!dominated) {
      kept.push(det);
    }
  }

  return kept;
}

/**
 * Preprocess an image (from video or canvas) for YOLOv8 inference.
 * Resizes to 640x640 with letterboxing, normalizes to [0, 1], converts to NCHW.
 * Returns the tensor data and scale/padding info for coordinate correction.
 */
function preprocess(
  source: HTMLVideoElement | HTMLCanvasElement
): {
  data: Float32Array;
  scaleX: number;
  scaleY: number;
  padX: number;
  padY: number;
} {
  const canvas = document.createElement("canvas");
  canvas.width = INPUT_SIZE;
  canvas.height = INPUT_SIZE;
  const ctx = canvas.getContext("2d")!;

  // Get source dimensions
  const srcW = source instanceof HTMLVideoElement ? source.videoWidth : source.width;
  const srcH = source instanceof HTMLVideoElement ? source.videoHeight : source.height;

  // Compute letterbox scaling (maintain aspect ratio)
  const scale = Math.min(INPUT_SIZE / srcW, INPUT_SIZE / srcH);
  const newW = Math.round(srcW * scale);
  const newH = Math.round(srcH * scale);
  const padX = (INPUT_SIZE - newW) / 2;
  const padY = (INPUT_SIZE - newH) / 2;

  // Fill with gray (114/255), then draw scaled image
  ctx.fillStyle = `rgb(114, 114, 114)`;
  ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  ctx.drawImage(source, padX, padY, newW, newH);

  // Get pixel data and convert to NCHW float32 normalized [0, 1]
  const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const pixels = imageData.data;
  const totalPixels = INPUT_SIZE * INPUT_SIZE;
  const data = new Float32Array(3 * totalPixels);

  for (let i = 0; i < totalPixels; i++) {
    const pi = i * 4;
    data[i] = pixels[pi] / 255;                    // R channel
    data[totalPixels + i] = pixels[pi + 1] / 255;  // G channel
    data[2 * totalPixels + i] = pixels[pi + 2] / 255; // B channel
  }

  return { data, scaleX: scale, scaleY: scale, padX, padY };
}

/**
 * Run YOLO detection on a video frame or canvas.
 * Returns detected key bounding boxes (a-z only) in original frame coordinates.
 */
export async function detectKeys(
  source: HTMLVideoElement | HTMLCanvasElement
): Promise<Record<string, [number, number, number, number]>> {
  if (!session || !ort) {
    throw new Error("YOLO not initialized. Call initYolo() first.");
  }

  // Preprocess
  const { data, scaleX, padX, padY } = preprocess(source);

  // Create input tensor [1, 3, 640, 640]
  const inputTensor = new ort.Tensor("float32", data, [1, 3, INPUT_SIZE, INPUT_SIZE]);

  // Run inference
  const feeds: Record<string, InstanceType<typeof ort.Tensor>> = {};
  feeds[session.inputNames[0]] = inputTensor;
  const results = await session.run(feeds);
  const output = results[session.outputNames[0]];

  // Output shape: [1, 64, 8400] — need to parse detections
  // Rows 0-3: cx, cy, w, h (in 640x640 letterboxed space)
  // Rows 4-63: class scores for 60 classes
  const outputData = output.data as Float32Array;
  const numDetections = 8400;
  const detections: YoloDetection[] = [];

  for (let i = 0; i < numDetections; i++) {
    // Find max class score
    let maxScore = 0;
    let maxClassIdx = 0;
    for (let c = 0; c < NUM_CLASSES; c++) {
      const score = outputData[(4 + c) * numDetections + i];
      if (score > maxScore) {
        maxScore = score;
        maxClassIdx = c;
      }
    }

    if (maxScore < CONF_THRESHOLD) continue;

    // Get raw label and apply fix
    let rawLabel = CLASS_NAMES[maxClassIdx] || "";
    rawLabel = YOLO_LABEL_FIX[rawLabel] ?? rawLabel;

    if (!isAllowedKey(rawLabel)) continue;

    // Extract bounding box (cx, cy, w, h) in letterboxed 640x640 space
    const cx = outputData[0 * numDetections + i];
    const cy = outputData[1 * numDetections + i];
    const bw = outputData[2 * numDetections + i];
    const bh = outputData[3 * numDetections + i];

    // Convert to x1, y1, x2, y2 in letterboxed space
    const lx1 = cx - bw / 2;
    const ly1 = cy - bh / 2;
    const lx2 = cx + bw / 2;
    const ly2 = cy + bh / 2;

    // Convert from letterboxed space back to original frame coordinates
    const x1 = (lx1 - padX) / scaleX;
    const y1 = (ly1 - padY) / scaleX; // scaleX === scaleY for letterbox
    const x2 = (lx2 - padX) / scaleX;
    const y2 = (ly2 - padY) / scaleX;

    detections.push({
      label: rawLabel,
      confidence: maxScore,
      x1: Math.round(x1),
      y1: Math.round(y1),
      x2: Math.round(x2),
      y2: Math.round(y2),
    });
  }

  // Apply NMS
  const filtered = nms(detections);

  // Keep highest confidence per key label
  const keyPositions: Record<string, [number, number, number, number]> = {};
  const bestConf: Record<string, number> = {};

  for (const det of filtered) {
    if (!bestConf[det.label] || det.confidence > bestConf[det.label]) {
      bestConf[det.label] = det.confidence;
      keyPositions[det.label] = [det.x1, det.y1, det.x2, det.y2];
    }
  }

  return keyPositions;
}

/**
 * Draw green bounding boxes on a canvas for detected keys (for annotated preview).
 */
export function annotateFrame(
  sourceCanvas: HTMLCanvasElement,
  keyPositions: Record<string, [number, number, number, number]>
): void {
  const ctx = sourceCanvas.getContext("2d");
  if (!ctx) return;

  ctx.strokeStyle = "rgb(0, 255, 0)";
  ctx.lineWidth = 3;
  ctx.font = "bold 14px sans-serif";
  ctx.fillStyle = "rgb(0, 255, 0)";

  for (const [label, [x1, y1, x2, y2]] of Object.entries(keyPositions)) {
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.fillText(label.toUpperCase(), x1 + 2, y1 - 4);
  }
}

/**
 * Clean up the YOLO session.
 */
export function disposeYolo(): void {
  try {
    if (session) {
      session.release();
      session = null;
    }
  } catch (err) {
    console.error("Error disposing YOLO detector:", err);
    session = null;
  }
}
