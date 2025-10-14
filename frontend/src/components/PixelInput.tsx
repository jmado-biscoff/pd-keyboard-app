import { cn } from "@/lib/utils";
import { InputHTMLAttributes, forwardRef } from "react";

interface PixelInputProps extends InputHTMLAttributes<HTMLInputElement> {}

export const PixelInput = forwardRef<HTMLInputElement, PixelInputProps>(
  ({ className, ...props }, ref) => {
    return (
      <input
        ref={ref}
        className={cn(
          "w-full px-4 py-3 bg-input border-[3px] border-border text-foreground font-pixel text-sm focus:outline-none focus:ring-2 focus:ring-primary transition-all",
          className
        )}
        {...props}
      />
    );
  }
);

PixelInput.displayName = "PixelInput";
