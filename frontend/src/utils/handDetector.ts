// MediaPipe Hand Landmark detection wrapper for browser
// Replaces Python MediaPipe + OpenCV hand tracking

// Lazy-loaded to avoid blocking module initialization
let mediapipe: typeof import("@mediapipe/tasks-vision") | null = null;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let handLandmarker: any = null;

export interface FingertipInfo {
  name: string;   // "thumb" | "index" | "middle" | "ring" | "pinky"
  x: number;      // pixel x coordinate
  y: number;      // pixel y coordinate
  hand: string;   // "left" | "right" (corrected for mirrored camera)
}

export interface HandDetectionResult {
  fingertips: FingertipInfo[];
  fingertipCount: number;
  leftFingersCount: number;
  rightFingersCount: number;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  landmarks: any;
}

const FINGERTIP_LANDMARKS: Record<string, number> = {
  thumb: 4,
  index: 8,
  middle: 12,
  ring: 16,
  pinky: 20,
};

/**
 * Initialize the MediaPipe HandLandmarker.
 * Downloads WASM + model files on first call.
 */
export async function initHandDetector(): Promise<void> {
  if (handLandmarker) return;

  // Dynamically import @mediapipe/tasks-vision (avoids blocking module load)
  if (!mediapipe) {
    mediapipe = await import("@mediapipe/tasks-vision");
  }

  const vision = await mediapipe.FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  handLandmarker = await mediapipe.HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.4,
    minTrackingConfidence: 0.4,
  });
}

/**
 * Detect hands in a video frame.
 * Returns fingertip positions with corrected handedness.
 */
export function detectHands(
  video: HTMLVideoElement,
  timestamp: number
): HandDetectionResult {
  if (!handLandmarker) {
    return { fingertips: [], fingertipCount: 0, leftFingersCount: 0, rightFingersCount: 0, landmarks: null };
  }

  const result = handLandmarker.detectForVideo(video, timestamp);
  const fingertips: FingertipInfo[] = [];
  let leftFingersCount = 0;
  let rightFingersCount = 0;

  const frameW = video.videoWidth;
  const frameH = video.videoHeight;

  if (result.landmarks && result.handednesses) {
    for (let i = 0; i < result.landmarks.length; i++) {
      const landmarks = result.landmarks[i];
      const handedness = result.handednesses[i];

      // Use MediaPipe's labels directly — browser getUserMedia provides a
      // mirrored (selfie) view, so MediaPipe's labels already match the
      // user's perspective without needing the swap that Python required.
      let hand: string;
      if (handedness && handedness.length > 0) {
        const label = handedness[0].categoryName;
        hand = label.toLowerCase();
      } else {
        // Fallback: use wrist x position
        hand = landmarks[0].x < 0.5 ? "left" : "right";
      }

      // Extract fingertip positions
      for (const [name, idx] of Object.entries(FINGERTIP_LANDMARKS)) {
        const lm = landmarks[idx];
        const x = Math.round(lm.x * frameW);
        const y = Math.round(lm.y * frameH);
        fingertips.push({ name, x, y, hand });

        if (hand === "left") leftFingersCount++;
        else rightFingersCount++;
      }
    }
  }

  return {
    fingertips,
    fingertipCount: fingertips.length,
    leftFingersCount,
    rightFingersCount,
    landmarks: result,
  };
}

/**
 * Clean up the HandLandmarker instance.
 */
export function disposeHandDetector(): void {
  try {
    if (handLandmarker) {
      handLandmarker.close();
      handLandmarker = null;
    }
  } catch (err) {
    console.error("Error disposing hand detector:", err);
    handLandmarker = null;
  }
}
