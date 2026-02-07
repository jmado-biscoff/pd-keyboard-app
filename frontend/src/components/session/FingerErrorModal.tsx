interface FingerErrorModalProps {
  show: boolean;
  leftFingersCount: number;
  rightFingersCount: number;
}

export const FingerErrorModal = ({
  show,
  leftFingersCount,
  rightFingersCount,
}: FingerErrorModalProps) => {
  if (!show) return null;

  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-card border-2 border-yellow-500 rounded-lg p-6 max-w-md shadow-2xl">
        <h2 className="text-xl font-pixel text-yellow-500 mb-4">
          ✋ Hand Position Required
        </h2>
        <p className="text-sm mb-4 text-foreground">
          Rest both hands properly on the home row keys.
        </p>

        {/* Dual Hand Status Boxes */}
        <div className="grid grid-cols-2 gap-3 mb-4">
          {/* Left Hand Status */}
          <div
            className={`p-4 rounded-lg border-2 ${
              leftFingersCount > 0
                ? "bg-green-500 border-green-600"
                : "bg-red-500 border-red-600"
            }`}
          >
            <div className="text-center">
              <div className="text-3xl mb-2">
                {leftFingersCount > 0 ? "✅" : "❌"}
              </div>
              <p className="font-pixel text-sm text-white">
                {leftFingersCount > 0 ? "Left hand detected" : "Left hand not detected"}
              </p>
            </div>
          </div>

          {/* Right Hand Status */}
          <div
            className={`p-4 rounded-lg border-2 ${
              rightFingersCount > 0
                ? "bg-green-500 border-green-600"
                : "bg-blue-500 border-blue-600"
            }`}
          >
            <div className="text-center">
              <div className="text-3xl mb-2">
                {rightFingersCount > 0 ? "✅" : "❌"}
              </div>
              <p className="font-pixel text-sm text-white">
                {rightFingersCount > 0 ? "Right hand detected" : "Right hand not detected"}
              </p>
            </div>
          </div>
        </div>

        <p className="text-xs text-muted-foreground text-center">
          Typing is blocked until hands are properly positioned.
          <br />
          You can still use Tab (Recalibrate) or Enter (Finish Session).
        </p>
      </div>
    </div>
  );
};
