import { useEffect, useRef } from "react";

interface VideoFeedProps {
  mode: "calibration" | "detection" | "idle";
  calibrationFrame?: string | null;
  videoRef?: React.RefObject<HTMLVideoElement>;
  canvasRef?: React.RefObject<HTMLCanvasElement>;
  calibrationDone?: boolean;
}

export const VideoFeed = ({
  mode,
  calibrationFrame,
  videoRef,
  canvasRef,
}: VideoFeedProps) => {
  const internalCanvasRef = useRef<HTMLCanvasElement>(null);

  // Render calibration frames (base64 JPEG from server)
  useEffect(() => {
    if (mode !== "calibration" || !calibrationFrame) return;

    const canvas = internalCanvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d", { alpha: false });
    if (!ctx) return;

    const renderFrame = async () => {
      try {
        const byteCharacters = atob(calibrationFrame);
        const byteNumbers = new Uint8Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const blob = new Blob([byteNumbers], { type: "image/jpeg" });
        const bitmap = await createImageBitmap(blob);

        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
        ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
        bitmap.close();
      } catch (err) {
        console.error("VideoFeed: Frame rendering error:", err);
      }
    };

    renderFrame();
  }, [mode, calibrationFrame]);

  return (
    <div className="w-full flex flex-col items-center">
      {/* Header with status */}
      <div className="flex items-center gap-2 mb-1">
        {mode !== "idle" ? (
          <>
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse shadow-[0_0_6px_rgba(220,60,60,0.6)]" />
            <p className="font-pixel text-[8px] uppercase tracking-widest text-muted-foreground/60">
              {mode === "calibration" ? "Calibrating" : "Live Feed"}
            </p>
          </>
        ) : (
          <p className="font-pixel text-[8px] uppercase tracking-widest text-muted-foreground/40">
            Camera
          </p>
        )}
      </div>

      {/* Video feed container */}
      <div
        className="w-full max-w-2xl rounded-lg border border-border/40 overflow-hidden bg-black/30 shadow-inner relative"
        style={{ aspectRatio: "16 / 9" }}
      >
        {mode === "calibration" ? (
          /* Calibration: render annotated base64 frames from server */
          <canvas
            ref={internalCanvasRef}
            className="w-full h-full object-contain"
            style={{ imageRendering: "auto", transform: "translateZ(0)" }}
          />
        ) : mode === "detection" ? (
          /* Detection: direct video feed + canvas overlay drawn by orchestrator */
          <>
            <video
              ref={videoRef}
              className="w-full h-full object-contain absolute inset-0"
              autoPlay
              playsInline
              muted
              style={{ transform: "translateZ(0)" }}
            />
            <canvas
              ref={canvasRef}
              className="w-full h-full object-contain absolute inset-0"
              style={{ imageRendering: "auto", transform: "translateZ(0)" }}
            />
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
