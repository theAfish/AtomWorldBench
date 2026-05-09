import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve, dirname } from 'path'
import { fileURLToPath } from 'url'

const __dirname = dirname(fileURLToPath(import.meta.url))

export default defineConfig({
  plugins: [react()],
  // Relative base so asset URLs work on GitHub Pages (/AtomWorldBench/) and bare domains.
  base: './',
  build: {
    outDir: '../docs',
    emptyOutDir: false,
    assetsDir: 'assets/fe',
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
      },
    },
  },
  server: {
    host: '0.0.0.0',
    port: 50001,
    allowedHosts: true,
  },
})
