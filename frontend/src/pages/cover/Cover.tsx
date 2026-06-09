import { Link } from 'react-router-dom'
import '../../styles/global.css'
import styles from './Cover.module.css'
import { TopBar } from '../../components/TopBar'

const CARDS: { title: string; desc: string; to?: string; href?: string; cta: string }[] = [
  {
    title: 'Documentation Hub',
    desc: 'Install, run CLI commands, inspect Python APIs, and understand benchmark actions.',
    to: '/docs',
    cta: 'Open documentation',
  },
  {
    title: 'Leaderboard',
    desc: 'Compare model success rates and error breakdowns by action across simple and verbose datasets.',
    to: '/dashboard',
    cta: 'View leaderboard',
  },
  {
    title: 'API Usage',
    desc: 'Step-by-step guide to creating a session, fetching tasks, submitting results, and retrieving scores via the REST API.',
    to: '/api-usage',
    cta: 'Read API guide',
  },
  {
    title: 'Agent Mode',
    desc: 'Run external tools or autonomous systems with clean task isolation and reproducible logs.',
    to: '/agent-mode',
    cta: 'Open agent mode docs',
  },
  {
    title: 'Project Source',
    desc: 'Inspect implementation details, run local workflows, and contribute improvements.',
    href: 'https://github.com/MasterAI-EAM/atomworld',
    cta: 'GitHub repository',
  },
]

export function Cover() {
  return (
    <div className={styles.page}>
      <TopBar />

      <section className={styles.hero}>
        <h1>AtomWorldBench</h1>
        <p>
          A benchmark for evaluating AI agents and language models on crystal-structure
          manipulation tasks. Explore the leaderboard, read the API guide, and start evaluating.
        </p>
        <div className={styles.badgeRow}>
          <span className={styles.badge}>crystal-structure benchmark</span>
          <span className={styles.badge}>agentic evaluation</span>
          <span className={styles.badge}>REST API</span>
        </div>
      </section>

      <section className={styles.grid}>
        {CARDS.map(({ title, desc, to, href, cta }) => (
          <article key={to ?? href} className={styles.card}>
            <h2>{title}</h2>
            <p>{desc}</p>
            {to
              ? <Link to={to}>{cta}</Link>
              : <a href={href} target="_blank" rel="noreferrer">{cta}</a>
            }
          </article>
        ))}
      </section>

      <footer className={styles.footer}>AtomWorldBench</footer>
    </div>
  )
}
