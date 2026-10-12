import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { '@': path.resolve(__dirname, './src') } },
  // Served from disk by the local app, so bundle size costs no download.
  build: { outDir: '../src/webui/dist', emptyOutDir: true, chunkSizeWarningLimit: 1024 },
  // `uv run x2loc-app --port 8765 --token dev --no-browser` serves the API.
  server: { proxy: { '/api': 'http://127.0.0.1:8765' } },
})
