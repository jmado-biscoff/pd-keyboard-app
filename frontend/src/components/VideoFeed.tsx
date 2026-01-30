import { useEffect, useRef, useState } from "react";

interface VideoFeedProps {
  detecting: boolean;
  calibrationDone: boolean;
  baseUrl: string;
  onFrame?: (frame: string) => void;
}

export const VideoFeed = ({ detecting, calibrationDone, baseUrl }: VideoFeedProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sseRef = useRef<EventSource | null>(null);
  const [fps, setFps] = useState(0);
  const frameTimesRef = useRef<number[]>([]);
  const firstFrameLoggedRef = useRef(false);
  const canvasInitializedRef = useRef(false);
  const lastFrameTimeRef = useRef<number>(Date.now());
  const firstFrameAfterCalibrationRef = useRef(false);

  useEffect(() => {
    if (!detecting) {
      // Reset state when detection stops
      if (sseRef.current) {
        sseRef.current.close();
        sseRef.current = null;
      }
      firstFrameLoggedRef.current = false;
      canvasInitializedRef.current = false;
      firstFrameAfterCalibrationRef.current = false;
      frameTimesRef.current = [];
      setFps(0);
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) {
      console.error("❌ VideoFeed: Canvas element not found");
      return;
    }

    const ctx = canvas.getContext("2d", {
      alpha: false,
      desynchronized: true, // Reduce latency for smoother rendering
      willReadFrequently: false, // Optimize for drawing, not reading
    });

    if (!ctx) {
      console.error("❌ VideoFeed: Failed to get 2D context");
      return;
    }

    console.log(`🎥 VideoFeed: Initializing SSE stream at ${baseUrl}/api/detect/stream`);
    const source = new EventSource(`${baseUrl}/api/detect/stream`);
    sseRef.current = source;

    source.onopen = () => {
      console.log("✅ VideoFeed: SSE connection established");
    };

    // ============================================================
    // CRITICAL: Unified frame rendering function
    // Renders frames continuously for main live feed
    // ============================================================
    const renderFrame = async (base64Frame: string) => {
      if (!canvas || !ctx) return;

      try {
        // Decode Base64 to binary
        const byteCharacters = atob(base64Frame);
        const byteNumbers = new Uint8Array(byteCharacters.length);

        // Fast typed array conversion
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }

        const blob = new Blob([byteNumbers], { type: "image/jpeg" });

        // Use createImageBitmap for hardware-accelerated decoding
        const bitmap = await createImageBitmap(blob);

        // Log first frame success
        if (!firstFrameLoggedRef.current) {
          console.log(`🎬 VideoFeed: First frame rendered (${bitmap.width}x${bitmap.height})`);
          firstFrameLoggedRef.current = true;
        }

        // Auto-resize canvas to match frame dimensions (only once)
        if (!canvasInitializedRef.current) {
          canvas.width = bitmap.width;
          canvas.height = bitmap.height;
          canvasInitializedRef.current = true;
          console.log(`📐 Canvas initialized to ${bitmap.width}x${bitmap.height}`);
        }

        // Draw frame directly without clearing (keeps last frame visible if gaps occur)
        ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
        bitmap.close(); // Free memory immediately

        // Update last frame timestamp
        lastFrameTimeRef.current = Date.now();

        // Track FPS continuously
        const now = performance.now();
        frameTimesRef.current.push(now);

        // Keep last 60 frames for rolling average
        if (frameTimesRef.current.length > 60) {
          frameTimesRef.current.shift();
        }

        // Calculate actual FPS
        if (frameTimesRef.current.length > 1) {
          const elapsed = (now - frameTimesRef.current[0]) / 1000;
          const currentFps = (frameTimesRef.current.length - 1) / elapsed;
          setFps(Math.round(currentFps));
        }
      } catch (err) {
        console.error("❌ VideoFeed: Frame rendering error:", err);
      }
    };

    source.onmessage = async (event: MessageEvent<string>) => {
      try {
        const data = JSON.parse(event.data);

        // ============================================================
        // Handle FRAME events (main detection loop)
        // ============================================================
        if (data.type === "frame" && data.frame) {
          if (calibrationDone && !firstFrameAfterCalibrationRef.current) {
            console.log("🎬 First frame after calibration received - resuming live feed");
            firstFrameAfterCalibrationRef.current = true;
          }
          await renderFrame(data.frame);
        }

        // ============================================================
        // Handle CALIBRATION_PROGRESS events (contains frames!)
        // ============================================================
        else if (data.type === "calibration_progress" && data.frame) {
          await renderFrame(data.frame);
          console.log(`🔄 Calibration: ${data.detected}/${data.required} keys detected`);
        }

        // ============================================================
        // Handle CALIBRATION_DONE event
        // ============================================================
        else if (data.type === "calibration_done") {
          console.log("✅ Calibration complete! Main live feed continues rendering.");
          console.log(`📡 Waiting for regular frame events (type: "frame")...`);
          // Reset frame tracking to ensure we detect any gaps
          lastFrameTimeRef.current = Date.now();
          firstFrameAfterCalibrationRef.current = false;
        }

        // ============================================================
        // Handle FPS stats from backend
        // ============================================================
        else if (data.type === "fps_stats") {
          console.log(
            `📊 Backend: Visual=${data.visual_fps} FPS, Inference=${data.inference_fps} FPS`
          );
        }
      } catch (err) {
        console.error("❌ VideoFeed: Message parsing error:", err);
      }
    };

    source.onerror = (event) => {
      console.error("❌ VideoFeed: SSE connection error", event);
      if (source.readyState === EventSource.CLOSED) {
        console.error("⚠️  SSE connection closed. EventSource will auto-reconnect...");
      }
    };

    // Monitor for frame reception gaps (console warning only)
    const frameCheckInterval = setInterval(() => {
      const timeSinceLastFrame = Date.now() - lastFrameTimeRef.current;
      if (timeSinceLastFrame > 5000 && detecting) {
        // No frames for 5+ seconds - log warning
        console.warn(`⚠️ VideoFeed: No frames received for ${timeSinceLastFrame}ms`);
      }
    }, 2000);

    return () => {
      console.log("🛑 VideoFeed: Cleaning up SSE connection");
      source.close();
      sseRef.current = null;
      clearInterval(frameCheckInterval);
    };
  }, [detecting, baseUrl]);

  return (
    <div className="w-full flex flex-col items-center">
      {/* Header with status and FPS */}
      <div className="flex items-center gap-2 mb-1">
        {detecting ? (
          <>
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse shadow-[0_0_6px_rgba(220,60,60,0.6)]" />
            <p className="font-pixel text-[8px] uppercase tracking-widest text-muted-foreground/60">
              {calibrationDone ? "Live Feed" : "Calibrating"}
            </p>
            {/* FPS indicator - always show when detecting */}
            {fps > 0 && (
              <span
                className={`font-pixel text-[8px] px-1.5 py-0.5 rounded ${
                  fps >= 50
                    ? "bg-green-500/20 text-green-400"
                    : fps >= 30
                    ? "bg-yellow-500/20 text-yellow-400"
                    : "bg-red-500/20 text-red-400"
                }`}
              >
                {fps} FPS
              </span>
            )}
          </>
        ) : (
          <p className="font-pixel text-[8px] uppercase tracking-widest text-muted-foreground/40">
            Camera
          </p>
        )}
      </div>

      {/* Video feed container */}
      <div
        className="w-full max-w-md rounded-lg border border-border/40 overflow-hidden bg-black/30 shadow-inner relative"
        style={{ aspectRatio: "16 / 9" }}
      >
        {detecting ? (
          <>
            {/* Canvas - ALWAYS visible when detecting (main live feed) */}
            <canvas
              ref={canvasRef}
              className="w-full h-full object-contain"
              style={{
                imageRendering: "auto",
                // Force hardware acceleration
                transform: "translateZ(0)",
                willChange: "transform",
              }}
            />

            {/* Low FPS warning overlay */}
            {fps > 0 && fps < 45 && (
              <div className="absolute top-2 right-2 bg-red-500/90 text-white px-2 py-1 rounded text-[8px] font-pixel">
                ⚠️ FPS DROP: {fps}
              </div>
            )}
          </>
        ) : (
          <div className="w-full h-full flex items-center justify-center text-muted-foreground/30">
            <p className="font-pixel text-[9px] uppercase tracking-widest">
              Camera feed will appear here
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
