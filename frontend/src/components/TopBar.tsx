import { Link } from 'react-router-dom'
import styles from './TopBar.module.css'

const NAV_LINKS = [
  { to: '/', label: 'Cover' },
  { to: '/docs', label: 'Docs' },
  { to: '/dashboard', label: 'Leaderboard' },
  { to: '/api-usage', label: 'API Usage' },
]

export function TopBar() {
  return (
    <header className={styles.topbar}>
      <Link className={styles.brand} to="/">
        <span className={styles.dot} />
        AtomWorldBench
      </Link>
      <nav className={styles.nav} aria-label="Primary">
        {NAV_LINKS.map(({ to, label }) => (
          <Link key={to} to={to}>
            {label}
          </Link>
        ))}
      </nav>
    </header>
  )
}
