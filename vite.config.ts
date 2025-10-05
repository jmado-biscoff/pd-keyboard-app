// vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  root: path.resolve(__dirname, "frontend"), // frontend folder with index.html
  build: {
    outDir: path.resolve(__dirname, "frontend/dist"), // frontend/dist folder
    emptyOutDir: true,
  },
});
