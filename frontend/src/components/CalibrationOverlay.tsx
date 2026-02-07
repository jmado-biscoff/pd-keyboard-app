import { PixelCard } from "./PixelCard";

interface CalibrationOverlayProps {
  isCalibrating: boolean;
  showCalibrationComplete: boolean;
  calibrationProgress: { detected: number; required: number };
}

export const CalibrationOverlay = ({
  isCalibrating,
  showCalibrationComplete,
  calibrationProgress,
}: CalibrationOverlayProps) => {
  if (!isCalibrating && !showCalibrationComplete) return null;

  const isCameraInitializing = calibrationProgress.detected === 0;

  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm z-50">
      <PixelCard className="p-8 text-center max-w-lg w-full mx-4 shadow-2xl">
        {isCalibrating ? (
          <>
            <p className="font-pixel text-lg text-yellow-400 mb-1 drop-shadow-[0_0_6px_rgba(244,169,66,0.4)]">
              {isCameraInitializing ? "Initializing Camera Hardware..." : "Auto-Calibrating... (26 Keys)"}
            </p>
            <p className="font-pixel text-xs text-muted-foreground/60 uppercase tracking-widest mb-5">
              {isCameraInitializing ? "Please wait..." : "Keep your keyboard in camera view"}
            </p>
            <div className="w-full bg-muted/50 rounded-full h-2.5 mb-2 overflow-hidden border border-border/40">
              <div
                className="bg-gradient-to-r from-yellow-400 via-yellow-300 to-yellow-400 h-full rounded-full transition-all duration-500 ease-out"
                style={{
                  width: `${(calibrationProgress.detected / calibrationProgress.required) * 100}%`,
                }}
              />
            </div>
            <p className="font-pixel text-xs text-muted-foreground/60 mb-5">
              {calibrationProgress.detected} <span className="text-muted-foreground/40">/</span> {calibrationProgress.required} keys detected
            </p>
          </>
        ) : (
          <p className="font-pixel text-lg text-green-400 drop-shadow-[0_0_8px_rgba(88,187,120,0.4)]">
            ✅ Calibration Complete!
          </p>
        )}
      </PixelCard>
    </div>
  );
};
