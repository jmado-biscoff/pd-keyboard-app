import { useMemo } from "react";
import { PixelCard } from "./PixelCard";
import { VirtualKeyboard } from "./VirtualKeyboard";

const ALL_KEYS = [
  "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P",
  "A", "S", "D", "F", "G", "H", "J", "K", "L",
  "Z", "X", "C", "V", "B", "N", "M",
];

interface CalibrationOverlayProps {
  isCalibrating: boolean;
  showCalibrationComplete: boolean;
  calibrationProgress: { detected: number; required: number };
  calibratedKeys?: string[];
}

export const CalibrationOverlay = ({
  isCalibrating,
  showCalibrationComplete,
  calibrationProgress,
  calibratedKeys = [],
}: CalibrationOverlayProps) => {
  if (!isCalibrating && !showCalibrationComplete) return null;

  const isCameraInitializing = calibrationProgress.detected === 0;

  // Build activeKeys map: all green when complete, only calibrated keys green during calibration
  const keyboardActiveKeys = useMemo(() => {
    if (showCalibrationComplete && !isCalibrating) {
      // All keys green on completion
      const allGreen: { [key: string]: string } = {};
      ALL_KEYS.forEach((k) => { allGreen[k] = "green"; });
      return allGreen;
    }
    // During calibration: light up detected keys
    const detected: { [key: string]: string } = {};
    calibratedKeys.forEach((k) => { detected[k] = "green"; });
    return detected;
  }, [isCalibrating, showCalibrationComplete, calibratedKeys]);

  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm z-50">
      <PixelCard className="p-8 text-center max-w-xl w-full mx-4 shadow-2xl">
        {isCalibrating ? (
          <>
            <p className="font-pixel text-lg text-yellow-400 mb-1 drop-shadow-[0_0_6px_rgba(244,169,66,0.4)]">
              {isCameraInitializing ? "Initializing..." : "Auto-Calibrating... (26 Keys)"}
            </p>

            {isCameraInitializing ? (
              // ✅ NEW: 2-Panel Layout with High Contrast
              <div className="space-y-4 mb-5 w-full max-w-lg mx-auto">
                <p className="font-pixel text-xs text-muted-foreground/80 uppercase tracking-widest">
                  Please wait...
                </p>

                {/* 2-Panel Grid */}
                <div className="grid grid-cols-2 gap-3">
                  {/* Panel 1: Remove Hands */}
                  <div className="bg-red-500/90 border-2 border-red-600 rounded-lg p-4 flex flex-col items-center justify-center text-center min-h-[120px]">
                    <div className="text-4xl mb-2">🚫</div>
                    <p className="font-pixel text-xs text-white font-bold mb-1">
                      Remove Hands
                    </p>
                    <p className="font-pixel text-[10px] text-white/90 leading-relaxed">
                      Take your hands off the keyboard now
                    </p>
                  </div>
                  {/* Panel 2: Keep Still */}
                  <div className="bg-blue-500/90 border-2 border-blue-600 rounded-lg p-4 flex flex-col items-center justify-center text-center min-h-[120px]">
                    <div className="text-4xl mb-2">🔒</div>
                    <p className="font-pixel text-xs text-white font-bold mb-1">
                      Keep Still
                    </p>
                    <p className="font-pixel text-[10px] text-white/90 leading-relaxed">
                      Don't move keyboard or camera
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              // Calibration phase (existing)
              <p className="font-pixel text-xs text-muted-foreground/60 uppercase tracking-widest mb-5">
                Keep your keyboard in camera view
              </p>
            )}

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

            {/* ✅ Only show keyboard during calibration, NOT during initialization */}
            {!isCameraInitializing && <VirtualKeyboard activeKeys={keyboardActiveKeys} />}
          </>
        ) : (
          <>
            <p className="font-pixel text-lg text-green-400 mb-4 drop-shadow-[0_0_8px_rgba(88,187,120,0.4)]">
              Calibration Complete
            </p>
            <VirtualKeyboard activeKeys={keyboardActiveKeys} />
          </>
        )}
      </PixelCard>
    </div>
  );
};
