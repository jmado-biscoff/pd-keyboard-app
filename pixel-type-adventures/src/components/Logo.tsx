import logoImg from "@/assets/TyPaw_logo.png";

export const Logo = ({ className = "" }: { className?: string }) => {
  return (
    <div className={`flex items-center justify-center ${className}`}>
      <img
        src={logoImg}
        alt="TypPaw Logo"
        className="h-28 w-auto select-none drop-shadow-[0_4px_4px_rgba(0,0,0,0.25)]"
      />
    </div>
  );
};
