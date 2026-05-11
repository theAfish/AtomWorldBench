/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Set to 'true' when building the static GitHub Pages site (no server). */
  readonly VITE_STATIC_SITE?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
