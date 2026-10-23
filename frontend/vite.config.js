import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Requests to /api/... are forwarded to the FastAPI server on port 8002.
// Example: /api/v1/ai/chat  ->  http://127.0.0.1:8002/api/v1/ai/chat
// This way the browser sees one origin and FastAPI needs no CORS setup.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8002",
        changeOrigin: true,
      },
    },
  },
});
