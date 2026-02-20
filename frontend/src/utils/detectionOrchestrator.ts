// Central coordinator for client-side typing detection
// Manages hand tracking loop + keystroke detection after calibration

import { detectHands, initHandDetector, disposeHandDetector, type FingertipInfo } from "./handDetector";
import { initSVM, predictKeystroke } from "./svmInference";
import { validateKeystroke } from "./groundTruth";

// Callback interface matching current SSE event types for minimal component changes
export interface DetectionCallbacks {
  onDetection: (data: {
    type: "detection";
    key: string;
    finger: string;
    hand: string;
    expected_key: string | null;
    ml_label: "Correct" | "Incorrect";
    fingertip_count: number;
    left_fingers_count: number;
    right_fingers_count: number;
  }) => void;
  onFrame: (data: {
    fingertip_count: number;
    left_fingers_count: number;
    right_fingers_count: number;
  }) => void;
  onError: (data: { message: string }) => void;
}

// Internal state
let isRunning = false;
let animFrameId: number | null = null;
let keyPositions: Record<string, number[]> = {};
let callbacks: DetectionCallbacks | null = null;
let videoElement: HTMLVideoElement | null = null;
let canvasElement: HTMLCanvasElement | null = null;
let mediaStream: MediaStream | null = null;
let lastFingertips: FingertipInfo[] = [];
let lastFingertipCount = 0;
let lastLeftCount = 0;
let lastRightCount = 0;
let lastKeystrokeTime = 0;
let keydownHandler: ((e: KeyboardEvent) => void) | null = null;

// Throttle hand detection to ~20 Hz
const DETECTION_INTERVAL_MS = 50;
let lastDetectionTime = 0;

// Debounce interval between keystrokes (same as Python: 250ms)
const DEBOUNCE_INTERVAL = 250;

// Minimum fingertips required for detection (lockdown mode, same as Python: 9)
const MIN_FINGERTIPS = 9;

/**
 * Initialize ML models (SVM + MediaPipe).
 * Call once before starting detection.
 */
export async function initModels(): Promise<void> {
  await Promise.all([initSVM(), initHandDetector()]);
}

/**
 * Start client-side detection after calibration.
 * Returns the MediaStream for cleanup.
 */
export async function startClientDetection(
  positions: Record<string, number[]>,
  cbs: DetectionCallbacks,
  videoRef: HTMLVideoElement,
  canvasRef: HTMLCanvasElement
): Promise<MediaStream> {
  if (isRunning) {
    stopClientDetection();
  }

  keyPositions = positions;
  callbacks = cbs;
  videoElement = videoRef;
  canvasElement = canvasRef;

  // Start camera
  mediaStream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 } },
  });

  videoElement.srcObject = mediaStream;
  await videoElement.play();

  isRunning = true;

  // Start hand tracking loop
  trackingLoop();

  // Attach keystroke handler to document
  keydownHandler = handleKeyDown;
  document.addEventListener("keydown", keydownHandler);

  return mediaStream;
}

/**
 * Stop client-side detection and clean up.
 * Does NOT dispose ML models — they persist for recalibration.
 */
export function stopClientDetection(): void {
  isRunning = false;

  if (animFrameId !== null) {
    cancelAnimationFrame(animFrameId);
    animFrameId = null;
  }

  if (keydownHandler) {
    document.removeEventListener("keydown", keydownHandler);
    keydownHandler = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }

  if (videoElement) {
    videoElement.srcObject = null;
    videoElement = null;
  }

  canvasElement = null;
  callbacks = null;
  lastFingertips = [];
  lastFingertipCount = 0;
  lastLeftCount = 0;
  lastRightCount = 0;
}

/**
 * Full cleanup: stop detection AND dispose ML models.
 * Call only on page unmount, not on recalibrate.
 */
export function disposeAll(): void {
  stopClientDetection();
  disposeHandDetector();
}

/**
 * Hand tracking loop running at ~20 Hz via requestAnimationFrame.
 */
function trackingLoop(): void {
  if (!isRunning || !videoElement || !canvasElement) return;

  const now = performance.now();

  if (now - lastDetectionTime >= DETECTION_INTERVAL_MS) {
    lastDetectionTime = now;

    if (videoElement.readyState >= 2) {
      const result = detectHands(videoElement, now);
      lastFingertips = result.fingertips;
      lastFingertipCount = result.fingertipCount;
      lastLeftCount = result.leftFingersCount;
      lastRightCount = result.rightFingersCount;

      // Draw overlays on canvas
      drawOverlay(result.fingertips);

      // Emit frame callback
      callbacks?.onFrame({
        fingertip_count: lastFingertipCount,
        left_fingers_count: lastLeftCount,
        right_fingers_count: lastRightCount,
      });
    }
  }

  animFrameId = requestAnimationFrame(trackingLoop);
}

/**
 * Draw hand landmarks and key bounding boxes on the canvas overlay.
 */
function drawOverlay(fingertips: FingertipInfo[]): void {
  if (!canvasElement || !videoElement) return;

  const ctx = canvasElement.getContext("2d");
  if (!ctx) return;

  const w = videoElement.videoWidth;
  const h = videoElement.videoHeight;
  canvasElement.width = w;
  canvasElement.height = h;

  ctx.clearRect(0, 0, w, h);

  // Draw key bounding boxes (cyan, same as Python post-calibration)
  ctx.strokeStyle = "rgb(0, 200, 200)";
  ctx.lineWidth = 3;
  for (const [, [x1, y1, x2, y2]] of Object.entries(keyPositions)) {
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }

  // Draw fingertip circles (yellow, same as Python)
  ctx.fillStyle = "rgb(0, 255, 255)";
  for (const ft of fingertips) {
    ctx.beginPath();
    ctx.arc(ft.x, ft.y, 6, 0, Math.PI * 2);
    ctx.fill();
  }

  // Draw hand indicators (L/R label over wrist area)
  // Group fingertips by hand to find approximate wrist position
  const hands = new Map<string, FingertipInfo[]>();
  for (const ft of fingertips) {
    const group = hands.get(ft.hand) || [];
    group.push(ft);
    hands.set(ft.hand, group);
  }

  ctx.font = "bold 20px sans-serif";
  ctx.textAlign = "center";
  for (const [hand, fts] of hands.entries()) {
    // Use average of all fingertips as approximate hand center
    const avgX = fts.reduce((s, f) => s + f.x, 0) / fts.length;
    const avgY = fts.reduce((s, f) => s + f.y, 0) / fts.length;
    const label = hand[0].toUpperCase();

    // Background box
    ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
    ctx.fillRect(avgX - 15, avgY - 40, 30, 28);
    ctx.strokeStyle = "rgb(0, 255, 0)";
    ctx.lineWidth = 2;
    ctx.strokeRect(avgX - 15, avgY - 40, 30, 28);

    // Label text
    ctx.fillStyle = "rgb(0, 255, 0)";
    ctx.fillText(label, avgX, avgY - 18);
  }
}

/**
 * Handle keydown events for keystroke detection.
 */
function handleKeyDown(e: KeyboardEvent): void {
  if (!isRunning || !callbacks) return;

  // Ignore held keys
  if (e.repeat) return;

  const key = e.key.toLowerCase();

  // Only process single letter keys (a-z)
  if (key.length !== 1 || !key.match(/[a-z]/)) return;

  // Debounce
  const now = Date.now();
  if (now - lastKeystrokeTime < DEBOUNCE_INTERVAL) return;
  lastKeystrokeTime = now;

  // Lockdown: skip if not enough fingertips visible
  if (lastFingertipCount < MIN_FINGERTIPS) return;

  // Find key's bounding box
  const bbox = keyPositions[key];
  if (!bbox || lastFingertips.length === 0) return;

  const [x1, y1, x2, y2] = bbox;
  const keyCx = (x1 + x2) / 2;
  const keyCy = (y1 + y2) / 2;

  // Find closest fingertip to key center (same as Python lines 438-448)
  let minDist = Infinity;
  let closestFinger: FingertipInfo | null = null;

  for (const ft of lastFingertips) {
    const dist = Math.sqrt((ft.x - keyCx) ** 2 + (ft.y - keyCy) ** 2);
    if (dist < minDist) {
      minDist = dist;
      closestFinger = ft;
    }
  }

  if (!closestFinger) return;

  // Compute SVM features
  const dx = closestFinger.x - keyCx;
  const dy = closestFinger.y - keyCy;
  const distance = Math.sqrt(dx * dx + dy * dy);
  const frameW = videoElement?.videoWidth || 1280;
  const frameH = videoElement?.videoHeight || 720;

  // Run SVM inference (async but fast — microseconds)
  predictKeystroke(key, closestFinger.name, closestFinger.hand, dx, dy, distance, frameW, frameH)
    .then((svmPred) => {
      if (!callbacks) return;

      // Apply ground truth validation
      const mlLabel = validateKeystroke(svmPred, closestFinger!.hand, closestFinger!.name, key);

      callbacks.onDetection({
        type: "detection",
        key: key.toUpperCase(),
        finger: closestFinger!.name,
        hand: closestFinger!.hand,
        expected_key: null, // Set by the component
        ml_label: mlLabel,
        fingertip_count: lastFingertipCount,
        left_fingers_count: lastLeftCount,
        right_fingers_count: lastRightCount,
      });
    })
    .catch((err) => {
      callbacks?.onError({ message: `SVM inference error: ${err.message}` });
    });
}
