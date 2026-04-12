import { PixelCard } from "./PixelCard";
import { PixelButton } from "./PixelButton";
import { AlertCircle } from "lucide-react";

interface LayoutErrorModalProps {
  show: boolean;
  onRecalibrate: () => void;
}

export const LayoutErrorModal = ({
  show,
  onRecalibrate,
}: LayoutErrorModalProps) => {
  if (!show) return null;

  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black/80 backdrop-blur-md z-[60]">
      <PixelCard className="p-10 text-center max-w-lg w-full mx-4 border-red-500/50 shadow-[0_0_50px_rgba(239,68,68,0.2)] animate-in fade-in zoom-in duration-300">
        <div className="flex justify-center mb-6">
          <div className="bg-red-500/20 p-4 rounded-full border border-red-500/50">
            <AlertCircle className="w-12 h-12 text-red-500" />
          </div>
        </div>
        
        <h2 className="font-pixel text-2xl text-red-500 mb-6 tracking-tight">
          Non-QWERTY Layout Detected
        </h2>

        <div className="bg-muted/30 border border-border/50 rounded-xl p-5 mb-8 text-center text-pixel">
          <p className="text-[11px] text-muted-foreground leading-relaxed">
            This system is optimized for <span className="text-yellow-400">QWERTY</span> keyboards.
            Please ensure your keyboard is visible and in QWERTY format.
          </p>
        </div>

        <div className="flex flex-col gap-3">
          <PixelButton variant="orange" className="w-full py-4 uppercase" onClick={onRecalibrate}>
            RECALIBRATE
          </PixelButton>
        </div>
      </PixelCard>
    </div>
  );
};
