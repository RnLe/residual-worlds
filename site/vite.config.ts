import { defineConfig } from "vite";

// Deployed under a project path by default; SITE_BASE overrides it for
// previews served from a different prefix.
export default defineConfig({
  base: process.env.SITE_BASE ?? "/residual-worlds/",
  build: { outDir: "dist" },
});
