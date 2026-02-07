import { PixelCard } from "./PixelCard";
import { PixelButton } from "./PixelButton";

interface DetectionErrorOverlayProps {
  detectionError: string | null;
  onRetry: () => void;
}

export const DetectionErrorOverlay = ({
  detectionError,
  onRetry,
}: DetectionErrorOverlayProps) => {
  if (!detectionError) return null;

  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black/70 z-50">
      <PixelCard className="p-8 text-center max-w-md">
        <p className="font-pixel text-lg text-red-500 mb-4">
          Keyboard Detection Error
        </p>
        <p className="font-pixel text-sm text-muted-foreground mb-6">
          {detectionError}
        </p>
        <PixelButton variant="primary" onClick={onRetry}>
          Retry Detection
        </PixelButton>
      </PixelCard>
    </div>
  );
};
