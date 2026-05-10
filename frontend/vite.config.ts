import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve, dirname } from 'path'
import { fileURLToPath } from 'url'

const __dirname = dirname(fileURLToPath(import.meta.url))

export default defineConfig({
  plugins: [react()],
  // Relative base so asset URLs work when served from any path.
  base: './',
  build: {
    outDir: './dist',
    emptyOutDir: true,
    assetsDir: 'assets',
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
      },
    },
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    allowedHosts: true,
    // Proxy API calls to the backend when running the Vite dev server.
    proxy: {
      '/auth': 'http://localhost:50001',
      '/sessions': 'http://localhost:50001',
      '/benchmark': 'http://localhost:50001',
      '/admin': 'http://localhost:50001',
      '/access-info': 'http://localhost:50001',
      '/healthz': 'http://localhost:50001',
    },
  },
})
