export function PixelButton(
  props: React.ButtonHTMLAttributes<HTMLButtonElement>
) {
  return (
    <button
      {...props}
      className={`px-4 py-2 bg-orange-500 text-white font-pixel rounded hover:bg-orange-600 ${
        props.className || ""
      }`}
    />
  );
}
