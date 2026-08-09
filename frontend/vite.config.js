import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The API base URL is injected at build time via VITE_API_URL so the same image
// can point at a local backend or a deployed one without a code change.
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: { '/api': { target: 'http://localhost:8000', changeOrigin: true, rewrite: (p) => p.replace(/^\/api/, '') } },
  },
  build: { outDir: 'dist', sourcemap: false },
});
