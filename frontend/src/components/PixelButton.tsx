import { cn } from "@/lib/utils";
import { ButtonHTMLAttributes, forwardRef } from "react";

interface PixelButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "accent" | "learn" | "play" | "settings";
  size?: "sm" | "md" | "lg" | "xl";
}

export const PixelButton = forwardRef<HTMLButtonElement, PixelButtonProps>(
  (
    { className, variant = "primary", size = "md", children, ...props },
    ref
  ) => {
    const variantClasses = {
      primary: "bg-primary text-primary-foreground hover:brightness-110",
      secondary: "bg-secondary text-secondary-foreground hover:brightness-110",
      accent: "bg-accent text-accent-foreground hover:brightness-110",
      learn: "bg-[#F4A942] text-foreground hover:brightness-110",
      play: "bg-[#FF8C42] text-white hover:brightness-110",
      settings: "bg-[#B24592] text-white hover:brightness-110",
    };

    const sizeClasses = {
      sm: "px-4 py-2 text-xs",
      md: "px-6 py-3 text-sm",
      lg: "px-8 py-4 text-base",
      xl: "px-12 py-6 text-lg",
    };

    return (
      <button
        ref={ref}
        className={cn(
          "pixel-border font-pixel uppercase transition-all active:translate-x-1 active:translate-y-1 active:shadow-none disabled:opacity-50 disabled:cursor-not-allowed",
          variantClasses[variant],
          sizeClasses[size],
          className
        )}
        {...props}
      >
        {children}
      </button>
    );
  }
);

PixelButton.displayName = "PixelButton";
