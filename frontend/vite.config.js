import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { copyFileSync } from 'fs';
var __dirname = dirname(fileURLToPath(import.meta.url));
// After the build, copy index.html → 404.html so GitHub Pages serves the
// SPA for any direct URL access (e.g. /atomworld/dashboard).  Without this,
// GitHub Pages returns its default 404 page because there is no matching
// static file at that path.
function ghPages404Plugin() {
    return {
        name: 'gh-pages-404',
        closeBundle: function () {
            var distDir = resolve(__dirname, 'dist');
            copyFileSync(resolve(distDir, 'index.html'), resolve(distDir, '404.html'));
        },
    };
}
export default defineConfig({
    plugins: [react(), ghPages404Plugin()],
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
});
