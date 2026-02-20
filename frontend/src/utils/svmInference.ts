// Client-side SVM ONNX inference for keystroke classification
// Replicates the Python encoder + scaler + SVM pipeline in the browser

// Lazy-loaded to avoid blocking module initialization
let ort: typeof import("onnxruntime-web") | null = null;

// Preprocessing data loaded from preprocessing.json
interface PreprocessingData {
  encoder: {
    categories: string[][];  // [pressed_key_cats, finger_name_cats, hand_used_cats]
  };
  scaler: {
    mean: number[];   // 6 values for [dx, dy, distance, norm_dx, norm_dy, norm_distance]
    scale: number[];  // 6 values
  };
}

let session: InstanceType<typeof import("onnxruntime-web").InferenceSession> | null = null;
let preprocessing: PreprocessingData | null = null;

/**
 * Initialize SVM model and preprocessing data.
 * Call once on app startup or before first inference.
 */
export async function initSVM(): Promise<void> {
  if (session && preprocessing) return;

  // Dynamically import onnxruntime-web (avoids blocking module load)
  if (!ort) {
    ort = await import("onnxruntime-web");
  }

  const prepResponse = await fetch("/models/preprocessing.json");

  if (!prepResponse.ok) {
    throw new Error("Failed to load preprocessing.json");
  }

  preprocessing = await prepResponse.json();
  session = await ort.InferenceSession.create("/models/svm_model.onnx");
}

/**
 * One-hot encode a categorical value given the category list.
 * Returns array of 0s with a 1 at the matching index, or all 0s if unknown.
 */
function oneHotEncode(value: string, categories: string[]): number[] {
  const encoded = new Array(categories.length).fill(0);
  const idx = categories.indexOf(value);
  if (idx !== -1) {
    encoded[idx] = 1;
  }
  return encoded;
}

/**
 * StandardScale a numeric value: (value - mean) / scale
 */
function standardScale(value: number, mean: number, scale: number): number {
  return (value - mean) / scale;
}

/**
 * Run SVM inference on a single keystroke.
 * Returns 0 (incorrect) or 1 (correct).
 */
export async function predictKeystroke(
  key: string,
  finger: string,
  hand: string,
  dx: number,
  dy: number,
  distance: number,
  frameW: number,
  frameH: number
): Promise<number> {
  if (!session || !preprocessing || !ort) {
    throw new Error("SVM not initialized. Call initSVM() first.");
  }

  const { encoder, scaler } = preprocessing;

  // Compute normalized features (same as Python)
  const normDx = dx / frameW;
  const normDy = dy / frameH;
  const normDistance = distance / Math.sqrt(frameW * frameW + frameH * frameH);

  // One-hot encode categorical features
  // encoder.categories order: [pressed_key, finger_name, hand_used]
  const encodedKey = oneHotEncode(key.toUpperCase(), encoder.categories[0]);
  const encodedFinger = oneHotEncode(finger.charAt(0).toUpperCase() + finger.slice(1).toLowerCase(), encoder.categories[1]);
  const encodedHand = oneHotEncode(hand.charAt(0).toUpperCase() + hand.slice(1).toLowerCase(), encoder.categories[2]);

  // Scale numeric features
  // scaler order: [dx, dy, distance, norm_dx, norm_dy, norm_distance]
  const scaledNumeric = [
    standardScale(dx, scaler.mean[0], scaler.scale[0]),
    standardScale(dy, scaler.mean[1], scaler.scale[1]),
    standardScale(distance, scaler.mean[2], scaler.scale[2]),
    standardScale(normDx, scaler.mean[3], scaler.scale[3]),
    standardScale(normDy, scaler.mean[4], scaler.scale[4]),
    standardScale(normDistance, scaler.mean[5], scaler.scale[5]),
  ];

  // Concatenate: [one-hot categorical] + [scaled numeric]
  const features = [...encodedKey, ...encodedFinger, ...encodedHand, ...scaledNumeric];

  // Create ONNX tensor and run inference
  const inputTensor = new ort.Tensor("float32", Float32Array.from(features), [1, features.length]);

  const feeds: Record<string, InstanceType<typeof ort.Tensor>> = {};
  const inputName = session.inputNames[0];
  feeds[inputName] = inputTensor;

  // Only request the label output (index 0), skip probability maps (index 1)
  // which are sequence-of-maps and not readable as tensors
  const labelName = session.outputNames[0];
  const results = await session.run(feeds, [labelName]);
  const output = results[labelName];

  // SVM output is the predicted class label (0 or 1)
  const prediction = (output.data as Float32Array | Int32Array | BigInt64Array)[0];
  return Number(prediction);
}
