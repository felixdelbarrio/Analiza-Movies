import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";

export default defineConfig({
  plugins: [react()],
  build: {
    chunkSizeWarningLimit: 600,
    rollupOptions: {
      output: {
        manualChunks: {
          react: [
            "react",
            "react-dom",
            "react-router-dom",
            "@tanstack/react-query",
            "@tanstack/react-virtual"
          ],
          motion: ["framer-motion", "lucide-react"],
          charts: [
            "echarts",
            "echarts/core",
            "echarts/charts",
            "echarts/components",
            "echarts/renderers",
            "echarts-for-react/lib/core"
          ]
        }
      }
    }
  },
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/health": "http://127.0.0.1:8000",
      "/ready": "http://127.0.0.1:8000",
      "/reports": "http://127.0.0.1:8000",
      "/cache": "http://127.0.0.1:8000",
      "/config": "http://127.0.0.1:8000",
      "/actions": "http://127.0.0.1:8000",
      "/meta": "http://127.0.0.1:8000"
    }
  },
  preview: {
    host: "0.0.0.0",
    port: 4173
  }
});
