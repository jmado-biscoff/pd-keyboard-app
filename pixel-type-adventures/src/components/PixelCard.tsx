import { cn } from "@/lib/utils";
import { HTMLAttributes } from "react";

interface PixelCardProps extends HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "orange" | "purple" | "yellow";
}

export const PixelCard = ({ className, variant = "default", children, ...props }: PixelCardProps) => {
  const variantClasses = {
    default: "bg-card",
    orange: "bg-[rgba(255,140,66,0.8)]",
    purple: "bg-[rgba(178,69,146,0.8)]",
    yellow: "bg-[rgba(244,169,66,0.8)]",
  };

  return (
    <div
      className={cn(
        "pixel-border p-6",
        variantClasses[variant],
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
};
