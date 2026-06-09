import { BrowserRouter, Routes, Route } from 'react-router-dom'

// basename is empty for local dev (served from /) and set to the deployment
// sub-path (e.g. "/atomworld") for GitHub Pages via VITE_ROUTER_BASENAME.
const basename = import.meta.env.VITE_ROUTER_BASENAME || ''

import { Cover } from './pages/cover/Cover'
import { Dashboard } from './pages/dashboard/Dashboard'
import { DocsPage } from './pages/docs/DocsPage'
import { RegisterPage } from './pages/register/RegisterPage'
import { AdminPage } from './pages/admin/AdminPage'

// Markdown files imported as raw strings at build time — no server needed.
import docsMd from './content/docs.md?raw'
import apiUsageMd from './content/api-usage.md?raw'
import installationMd from './content/installation.md?raw'
import pythonApiMd from './content/python-api.md?raw'
import actionsMd from './content/actions.md?raw'
import cliMd from './content/cli.md?raw'
import agentModeMd from './content/agent-mode.md?raw'
import dataGenMd from './content/data-generation.md?raw'
import contributingMd from './content/contributing.md?raw'
import citationMd from './content/citation.md?raw'

export function App() {
  return (
    <BrowserRouter basename={basename}>
      <Routes>
        <Route path="/" element={<Cover />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/docs" element={<DocsPage content={docsMd} title="Documentation" />} />
        <Route path="/api-usage" element={<DocsPage content={apiUsageMd} title="API Usage" />} />
        <Route path="/installation" element={<DocsPage content={installationMd} title="Installation" />} />
        <Route path="/python-api" element={<DocsPage content={pythonApiMd} title="Python API" />} />
        <Route path="/actions" element={<DocsPage content={actionsMd} title="Actions" />} />
        <Route path="/cli" element={<DocsPage content={cliMd} title="CLI" />} />
        <Route path="/agent-mode" element={<DocsPage content={agentModeMd} title="Agent Mode" />} />
        <Route path="/data-generation" element={<DocsPage content={dataGenMd} title="Data Generation" />} />
        <Route path="/contributing" element={<DocsPage content={contributingMd} title="Contributing" />} />
        <Route path="/citation" element={<DocsPage content={citationMd} title="Citation" />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route path="/admin" element={<AdminPage />} />
        {/* Fallback */}
        <Route path="*" element={<Cover />} />
      </Routes>
    </BrowserRouter>
  )
}
