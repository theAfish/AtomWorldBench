import { useState, useCallback } from 'react'
import { TopBar } from '../../components/TopBar'
import '../../styles/global.css'
import styles from './AdminPage.module.css'

interface UserRecord {
  username: string
  email?: string
  organization?: string
  created_at: string
}

interface KeyRecord {
  api_key: string
  username: string
  note?: string
  created_at: string
}

export function AdminPage() {
  const [adminKey, setAdminKey] = useState('')
  const [authed, setAuthed] = useState(false)
  const [authError, setAuthError] = useState<string | null>(null)
  const [authLoading, setAuthLoading] = useState(false)

  const [users, setUsers] = useState<UserRecord[]>([])
  const [keys, setKeys] = useState<KeyRecord[]>([])
  const [dataLoading, setDataLoading] = useState(false)
  const [dataError, setDataError] = useState<string | null>(null)

  // Issue key state
  const [issueUsername, setIssueUsername] = useState('')
  const [issueNote, setIssueNote] = useState('')
  const [issuedKey, setIssuedKey] = useState<string | null>(null)
  const [issueError, setIssueError] = useState<string | null>(null)
  const [issueLoading, setIssueLoading] = useState(false)

  const loadData = useCallback(async (key: string) => {
    setDataLoading(true)
    setDataError(null)
    try {
      const [usersResp, keysResp] = await Promise.all([
        fetch('/admin/users', { headers: { 'X-API-Key': key } }),
        fetch('/admin/keys', { headers: { 'X-API-Key': key } }),
      ])
      if (!usersResp.ok || !keysResp.ok) {
        const body = await (usersResp.ok ? keysResp : usersResp)
          .json()
          .catch(() => ({ detail: 'Failed to load data' }))
        throw new Error(body.detail ?? 'Failed to load admin data')
      }
      setUsers(await usersResp.json())
      setKeys(await keysResp.json())
    } catch (err: unknown) {
      setDataError(err instanceof Error ? err.message : String(err))
    } finally {
      setDataLoading(false)
    }
  }, [])

  async function handleAuth(e: React.FormEvent) {
    e.preventDefault()
    setAuthError(null)
    setAuthLoading(true)
    try {
      const resp = await fetch('/admin/users', {
        headers: { 'X-API-Key': adminKey },
      })
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({ detail: resp.statusText }))
        throw new Error(body.detail ?? `HTTP ${resp.status}`)
      }
      const usersData: UserRecord[] = await resp.json()
      const keysResp = await fetch('/admin/keys', {
        headers: { 'X-API-Key': adminKey },
      })
      setUsers(usersData)
      setKeys(keysResp.ok ? await keysResp.json() : [])
      setAuthed(true)
    } catch (err: unknown) {
      setAuthError(err instanceof Error ? err.message : String(err))
    } finally {
      setAuthLoading(false)
    }
  }

  async function handleIssueKey(e: React.FormEvent) {
    e.preventDefault()
    setIssueError(null)
    setIssuedKey(null)
    setIssueLoading(true)
    try {
      const resp = await fetch('/auth/issue-key', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': adminKey,
        },
        body: JSON.stringify({
          username: issueUsername.trim(),
          note: issueNote.trim() || null,
        }),
      })
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({ detail: resp.statusText }))
        throw new Error(body.detail ?? `HTTP ${resp.status}`)
      }
      const data = await resp.json()
      setIssuedKey(data.api_key)
      setIssueUsername('')
      setIssueNote('')
      // Refresh keys list
      loadData(adminKey)
    } catch (err: unknown) {
      setIssueError(err instanceof Error ? err.message : String(err))
    } finally {
      setIssueLoading(false)
    }
  }

  function fmtDate(iso: string) {
    return new Date(iso).toLocaleString()
  }

  return (
    <div className={styles.page}>
      <TopBar />

      <section className={styles.hero}>
        <h1>Admin Panel</h1>
        <p>Manage registered users and issued API keys. Requires the bootstrap admin key.</p>
      </section>

      {!authed ? (
        <div className={styles.authGate}>
          <h2>Enter admin key</h2>
          <form onSubmit={handleAuth}>
            <div className={styles.keyRow}>
              <input
                className={styles.keyInput}
                type="password"
                placeholder="Bootstrap admin API key"
                value={adminKey}
                onChange={e => setAdminKey(e.target.value)}
                autoComplete="current-password"
              />
              <button
                type="submit"
                className={styles.loadBtn}
                disabled={authLoading || !adminKey.trim()}
              >
                {authLoading ? 'Verifying…' : 'Load'}
              </button>
            </div>
            {authError && <div className={styles.errorMsg}>{authError}</div>}
          </form>
        </div>
      ) : (
        <div className={styles.panels}>
          {/* ── Issue key ─────────────────────────────────────────── */}
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <h2>Issue API Key</h2>
            </div>
            <form className={styles.issueForm} onSubmit={handleIssueKey}>
              <input
                className={styles.issueInput}
                type="text"
                placeholder="Username"
                value={issueUsername}
                onChange={e => setIssueUsername(e.target.value)}
                required
              />
              <input
                className={styles.issueInput}
                type="text"
                placeholder="Note (optional)"
                value={issueNote}
                onChange={e => setIssueNote(e.target.value)}
              />
              <button
                type="submit"
                className={styles.issueBtn}
                disabled={issueLoading || !issueUsername.trim()}
              >
                {issueLoading ? 'Issuing…' : 'Issue Key'}
              </button>
            </form>
            {issueError && <div className={styles.errorMsg}>{issueError}</div>}
            {issuedKey && (
              <div className={styles.issuedKey}>New key: {issuedKey}</div>
            )}
          </div>

          {/* ── Users table ───────────────────────────────────────── */}
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <h2>Registered Users ({users.length})</h2>
              <button
                className={styles.reloadBtn}
                onClick={() => loadData(adminKey)}
                disabled={dataLoading}
              >
                {dataLoading ? 'Loading…' : 'Refresh'}
              </button>
            </div>
            {dataError && <div className={styles.errorMsg}>{dataError}</div>}
            <div className={styles.tableWrap}>
              {users.length === 0 ? (
                <div className={styles.empty}>No users registered yet.</div>
              ) : (
                <table>
                  <thead>
                    <tr>
                      <th>Username</th>
                      <th>Email</th>
                      <th>Organization</th>
                      <th>Registered</th>
                    </tr>
                  </thead>
                  <tbody>
                    {users.map(u => (
                      <tr key={u.username}>
                        <td><strong>{u.username}</strong></td>
                        <td>{u.email ?? <span className={styles.badge}>—</span>}</td>
                        <td>{u.organization ?? <span className={styles.badge}>—</span>}</td>
                        <td className={styles.mono}>{fmtDate(u.created_at)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>

          {/* ── Keys table ────────────────────────────────────────── */}
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <h2>Issued API Keys ({keys.length})</h2>
            </div>
            <div className={styles.tableWrap}>
              {keys.length === 0 ? (
                <div className={styles.empty}>No keys issued yet.</div>
              ) : (
                <table>
                  <thead>
                    <tr>
                      <th>Key</th>
                      <th>Username</th>
                      <th>Note</th>
                      <th>Issued</th>
                    </tr>
                  </thead>
                  <tbody>
                    {keys.map(k => (
                      <tr key={k.api_key}>
                        <td className={styles.mono}>{k.api_key}</td>
                        <td>{k.username}</td>
                        <td>{k.note ?? <span className={styles.badge}>—</span>}</td>
                        <td className={styles.mono}>{fmtDate(k.created_at)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
