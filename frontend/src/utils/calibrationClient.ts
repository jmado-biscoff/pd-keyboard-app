// Browser-side calibration manager
// Captures frames from the user's camera and runs YOLO detection locally
// All ML inference happens in the browser — no server round-trips needed

import { initYolo, detectKeys, annotateFrame, disposeYolo } from "./yoloDetector";

export interface CalibrationProgress {
  detected: number;
  required: number;
  detected_keys: string[];
  annotated_frame: string | null;
}

interface CalibrationConfig {
  pollIntervalMs?: number;   // How often to run detection (default: 500ms)
  timeoutMs?: number;        // Max time before giving up (default: 30000ms)
  motionThreshold?: number;  // Max center movement before reset (default: 20px)
  stuckTimeoutMs?: number;   // Reset if no progress for this long (default: 4000ms)
}

const DEFAULT_CONFIG: Required<CalibrationConfig> = {
  pollIntervalMs: 500,
  timeoutMs: 30000,
  motionThreshold: 20,
  stuckTimeoutMs: 4000,
};

/**
 * Run calibration using client-side YOLO inference.
 * Captures frames from the webcam, runs YOLO locally, and accumulates key positions.
 */
export async function runCalibration(
  videoElement: HTMLVideoElement,
  onProgress: (data: CalibrationProgress) => void,
  onComplete: (keyPositions: Record<string, number[]>) => void,
  onError: (msg: string) => void,
  config?: CalibrationConfig
): Promise<{ stop: () => void; stream: MediaStream }> {
  const cfg = { ...DEFAULT_CONFIG, ...config };

  // Initialize YOLO model (downloads ~44 MB on first call, cached after)
  try {
    await initYolo();
  } catch (err) {
    onError(`Failed to load YOLO model: ${err instanceof Error ? err.message : err}`);
    throw err;
  }

  // Start camera
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 } },
  });

  videoElement.srcObject = stream;
  await videoElement.play();

  // Canvas for capturing frames and drawing annotations
  const captureCanvas = document.createElement("canvas");

  let stopped = false;
  let intervalId: ReturnType<typeof setInterval> | null = null;

  // Accumulated best bounding boxes per key
  const bestPositions: Record<string, number[]> = {};
  let previousCenter: [number, number] | null = null;
  let lastProgressCount = 0;
  let lastProgressTime = Date.now();

  const stop = () => {
    stopped = true;
    if (intervalId) {
      clearInterval(intervalId);
      intervalId = null;
    }
  };

  // Overall timeout
  const overallTimeout = setTimeout(() => {
    if (!stopped) {
      stop();
      onError("Calibration timed out. Make sure your keyboard is visible to the camera.");
    }
  }, cfg.timeoutMs);

  const pollFrame = async () => {
    if (stopped || videoElement.readyState < 2) return;

    try {
      // Run YOLO detection directly on the video element
      const detections = await detectKeys(videoElement);

      // Merge detected keys into best positions
      for (const [key, bbox] of Object.entries(detections)) {
        bestPositions[key] = bbox as number[];
      }

      // Motion detection: reset if keyboard center moves too much
      if (Object.keys(bestPositions).length > 0) {
        const centers = Object.values(bestPositions).map(
          ([x1, y1, x2, y2]: number[]) => [(x1 + x2) / 2, (y1 + y2) / 2] as [number, number]
        );
        const avgX = centers.reduce((s, c) => s + c[0], 0) / centers.length;
        const avgY = centers.reduce((s, c) => s + c[1], 0) / centers.length;
        const currentCenter: [number, number] = [avgX, avgY];

        if (previousCenter) {
          const dx = Math.abs(currentCenter[0] - previousCenter[0]);
          const dy = Math.abs(currentCenter[1] - previousCenter[1]);
          const motionDist = Math.sqrt(dx * dx + dy * dy);

          if (motionDist > cfg.motionThreshold) {
            // Keyboard moved - reset
            for (const key of Object.keys(bestPositions)) {
              delete bestPositions[key];
            }
            previousCenter = null;
            lastProgressCount = 0;
            lastProgressTime = Date.now();
          }
        }
        previousCenter = currentCenter;
      }

      // Stuck detection: reset if no new keys for stuckTimeoutMs
      const currentCount = Object.keys(bestPositions).length;
      if (currentCount > lastProgressCount) {
        lastProgressCount = currentCount;
        lastProgressTime = Date.now();
      } else if (currentCount > 0 && currentCount < 26) {
        if (Date.now() - lastProgressTime > cfg.stuckTimeoutMs) {
          // Stuck - reset
          for (const key of Object.keys(bestPositions)) {
            delete bestPositions[key];
          }
          previousCenter = null;
          lastProgressCount = 0;
          lastProgressTime = Date.now();
        }
      }

      // Create annotated frame for UI preview
      let annotatedFrame: string | null = null;
      if (Object.keys(bestPositions).length > 0) {
        captureCanvas.width = videoElement.videoWidth;
        captureCanvas.height = videoElement.videoHeight;
        const ctx = captureCanvas.getContext("2d");
        if (ctx) {
          ctx.drawImage(videoElement, 0, 0);
          annotateFrame(captureCanvas, bestPositions as Record<string, [number, number, number, number]>);
          annotatedFrame = captureCanvas.toDataURL("image/jpeg", 0.75);
        }
      }

      // Report progress
      onProgress({
        detected: Object.keys(bestPositions).length,
        required: 26,
        detected_keys: Object.keys(bestPositions).sort(),
        annotated_frame: annotatedFrame,
      });

      // Check completion
      if (Object.keys(bestPositions).length >= 26) {
        clearTimeout(overallTimeout);
        stop();
        onComplete({ ...bestPositions });
      }
    } catch (err) {
      console.error("[Calibration] Detection error:", err);
    }
  };

  // Start polling
  intervalId = setInterval(pollFrame, cfg.pollIntervalMs);
  // Run first frame immediately
  pollFrame();

  return { stop, stream };
}

/**
 * Clean up YOLO resources after calibration.
 */
export function disposeCalibration(): void {
  disposeYolo();
}
