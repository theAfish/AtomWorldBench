import { useState } from 'react'
import { TopBar } from '../../components/TopBar'
import '../../styles/global.css'
import styles from './RegisterPage.module.css'

interface RegisterResult {
  username: string
  email?: string
  organization?: string
  api_key: string
  created_at: string
}

export function RegisterPage() {
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [organization, setOrganization] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<RegisterResult | null>(null)
  const [copied, setCopied] = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      const resp = await fetch('/auth/self-register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username: username.trim(),
          email: email.trim() || null,
          organization: organization.trim() || null,
        }),
      })
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({ detail: resp.statusText }))
        throw new Error(body.detail ?? `HTTP ${resp.status}`)
      }
      const data: RegisterResult = await resp.json()
      setResult(data)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  async function handleCopy() {
    if (!result) return
    await navigator.clipboard.writeText(result.api_key)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className={styles.page}>
      <TopBar />

      <section className={styles.hero}>
        <h1>Get an API Key</h1>
        <p>
          Register once to receive an API key. Use it with the{' '}
          <code>X-API-Key</code> header for all benchmark requests.
        </p>
      </section>

      <div className={styles.layout}>
        {/* ── Registration form ─────────────────────────────────────── */}
        <div className={styles.card}>
          <h2>Register</h2>

          {!result ? (
            <form className={styles.form} onSubmit={handleSubmit}>
              <div className={styles.fieldGroup}>
                <label htmlFor="reg-username">Username</label>
                <input
                  id="reg-username"
                  type="text"
                  required
                  placeholder="e.g. alice"
                  value={username}
                  onChange={e => setUsername(e.target.value)}
                  autoComplete="username"
                />
              </div>

              <div className={styles.fieldGroup}>
                <label htmlFor="reg-email">
                  Email <span className={styles.optional}>(optional)</span>
                </label>
                <input
                  id="reg-email"
                  type="email"
                  placeholder="alice@example.com"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  autoComplete="email"
                />
              </div>

              <div className={styles.fieldGroup}>
                <label htmlFor="reg-org">
                  Organization <span className={styles.optional}>(optional)</span>
                </label>
                <input
                  id="reg-org"
                  type="text"
                  placeholder="e.g. WMML"
                  value={organization}
                  onChange={e => setOrganization(e.target.value)}
                />
              </div>

              {error && <div className={styles.errorMsg}>{error}</div>}

              <button
                type="submit"
                className={styles.submitBtn}
                disabled={loading || !username.trim()}
              >
                {loading ? 'Registering…' : 'Register & Get Key'}
              </button>
            </form>
          ) : (
            <div className={styles.successBox}>
              <div className={styles.successTitle}>Registration successful!</div>

              <div className={styles.keyBox}>
                <div className={styles.keyLabel}>Your API Key</div>
                <div className={styles.keyValue}>{result.api_key}</div>
              </div>

              <button className={styles.copyBtn} onClick={handleCopy}>
                {copied ? 'Copied!' : 'Copy key'}
              </button>

              <div className={styles.warning}>
                Save this key — it will not be shown again. Use it in the{' '}
                <code>X-API-Key</code> header for every benchmark request.
              </div>

              <button
                className={styles.resetBtn}
                onClick={() => {
                  setResult(null)
                  setUsername('')
                  setEmail('')
                  setOrganization('')
                  setError(null)
                }}
              >
                Register another user
              </button>
            </div>
          )}
        </div>

        {/* ── Quick-start info ──────────────────────────────────────── */}
        <div className={styles.infoCard}>
          <h2>Quick start</h2>
          <ol className={styles.stepList}>
            <li>
              <span className={styles.stepNum}>1</span>
              <span className={styles.stepText}>
                Register here and save your <code>api_key</code>.
              </span>
            </li>
            <li>
              <span className={styles.stepNum}>2</span>
              <span className={styles.stepText}>
                Call <code>POST /benchmark</code> with your key and optional filters
                to receive all tasks in one response.
              </span>
            </li>
            <li>
              <span className={styles.stepNum}>3</span>
              <span className={styles.stepText}>
                For each task, read <code>action_prompt</code> and{' '}
                <code>input_cif</code>, generate a result CIF, and submit with{' '}
                <code>POST /sessions/{'{session_id}'}/tasks/{'{task_id}'}/submit</code>.
              </span>
            </li>
            <li>
              <span className={styles.stepNum}>4</span>
              <span className={styles.stepText}>
                After all submissions, call <code>POST /sessions/{'{session_id}'}/evaluate</code>{' '}
                and then <code>GET /sessions/{'{session_id}'}/results</code>.
              </span>
            </li>
          </ol>
        </div>
      </div>
    </div>
  )
}
