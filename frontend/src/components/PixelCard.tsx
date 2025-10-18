import { cn } from "@/lib/utils";
import { HTMLAttributes } from "react";

interface PixelCardProps extends HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "orange" | "red" | "yellow" | "purple" | "green";
  translucent?: boolean;
}

export const PixelCard = ({
  className,
  variant = "default",
  translucent = false,
  children,
  ...props
}: PixelCardProps) => {
  const variantClasses: Record<string, string> = {
    default: "bg-card",
    orange: "bg-[rgba(255,140,66,0.8)]",
    red: "bg-[rgba(220,60,60,0.85)]",
    yellow: "bg-[rgba(244,169,66,0.8)]",
    purple: "bg-[rgba(178,69,146,0.8)]",
    green: "bg-[rgba(88,187,120,0.8)]",
  };

  const translucentClass = "bg-[rgba(20,20,20,0.45)] backdrop-blur-sm";

  const bgClass = translucent ? translucentClass : variantClasses[variant];

  return (
    <div
      className={cn("pixel-border p-6", bgClass, className)}
      {...props}
    >
      {children}
    </div>
  );
};
