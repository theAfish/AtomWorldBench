import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import '../../styles/global.css'
import styles from './Dashboard.module.css'
import type { BenchmarkData } from '../../types'
import { TopBar } from '../../components/TopBar'
import { SRHeatmap } from './charts/SRHeatmap'
import { MMDHeatmap } from './charts/MMDHeatmap'
import { BreakdownChart } from './charts/BreakdownChart'

type Dataset = 'simple' | 'verbose'
type Tab = 'heatmaps' | 'breakdown'

async function loadDataset(type: Dataset): Promise<BenchmarkData> {
  const resp = await fetch(`data/${type}_metrics.json`)
  if (!resp.ok) throw new Error(`Failed to load ${type}_metrics.json: ${resp.statusText}`)
  return resp.json() as Promise<BenchmarkData>
}

export function Dashboard() {
  const [dataset, setDataset] = useState<Dataset>('simple')
  const [data, setData] = useState<BenchmarkData | null>(null)
  const [activeTab, setActiveTab] = useState<Tab>('heatmaps')
  const [activeAction, setActiveAction] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    loadDataset(dataset)
      .then(d => {
        setData(d)
        setActiveAction(prev =>
          prev && d.actions.includes(prev) ? prev : d.actions[0] ?? null,
        )
        setLoading(false)
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : String(err))
        setLoading(false)
      })
  }, [dataset])

  return (
    <div className={styles.page}>
      <TopBar />

      <section className={styles.hero}>
        <div className={styles.heroText}>
          <h1>Leaderboard</h1>
          <p>Benchmark performance overview for crystal-structure manipulation tasks</p>
        </div>
        <div className={styles.controls}>
          <div className={styles.controlGroup}>
            <span className={styles.controlLabel}>Dataset</span>
            {(['simple', 'verbose'] as const).map(d => (
              <button
                key={d}
                className={`${styles.pill} ${dataset === d ? styles.pillActive : ''}`}
                onClick={() => setDataset(d)}
              >
                {d === 'simple' ? 'Simple' : 'Verbose'}
              </button>
            ))}
          </div>
          <div className={styles.controlGroup}>
            <span className={styles.controlLabel}>View</span>
            {(['heatmaps', 'breakdown'] as const).map(tab => (
              <button
                key={tab}
                className={`${styles.pill} ${activeTab === tab ? styles.pillActive : ''}`}
                onClick={() => setActiveTab(tab)}
              >
                {tab === 'heatmaps' ? 'Heatmaps' : 'Breakdown'}
              </button>
            ))}
          </div>
        </div>
      </section>

      <main className={styles.main}>
        {loading && (
          <div className={styles.loading}>
            <div className={styles.spinner} />
            Loading results…
          </div>
        )}

        {error && <div className={styles.errorMsg}>⚠ {error}</div>}

        {!loading && !error && data && (
          <>
            {/* Heatmaps panel */}
            <div className={activeTab === 'heatmaps' ? styles.panelActive : styles.panel}>
              <div className={styles.grid2}>
                <div className={styles.card}>
                  <h2>Success Rate</h2>
                  <p className={styles.subtitle}>
                    Fraction of correct predictions (model × action, N = 250 samples)
                  </p>
                  <SRHeatmap data={data} visible={activeTab === 'heatmaps'} />
                </div>
                <div className={styles.card}>
                  <h2>Mean Max-Distance of Correct Predictions</h2>
                  <p className={styles.subtitle}>
                    Lower is better — average maximum atom displacement (Å) over correct
                    predictions only
                  </p>
                  <MMDHeatmap data={data} visible={activeTab === 'heatmaps'} />
                </div>
              </div>
            </div>

            {/* Breakdown panel */}
            <div className={activeTab === 'breakdown' ? styles.panelActive : styles.panel}>
              <div className={styles.card}>
                <h2>Outcome Breakdown per Action</h2>
                <p className={styles.subtitle}>
                  Stacked bars show the fraction of each result type across models
                </p>
                <div className={styles.legend}>
                  {[
                    { label: 'Correct', color: '#3d9970' },
                    { label: 'StructureMismatch', color: '#ff851b' },
                    { label: 'AtomCountMismatch', color: '#4c78a8' },
                    { label: 'CIFParsingError', color: '#e15759' },
                    { label: 'OutputFormatError', color: '#9467bd' },
                    { label: 'Other', color: '#aec7e8' },
                  ].map(({ label, color }) => (
                    <div key={label} className={styles.legendItem}>
                      <div className={styles.legendDot} style={{ background: color }} />
                      {label}
                    </div>
                  ))}
                </div>
                <div className={styles.actionTabs}>
                  {data.actions.map(a => (
                    <button
                      key={a}
                      className={`${styles.pill} ${activeAction === a ? styles.pillActive : ''}`}
                      onClick={() => setActiveAction(a)}
                    >
                      {data.action_labels[a] ?? a}
                    </button>
                  ))}
                </div>
                {activeAction && (
                  <BreakdownChart
                    data={data}
                    actionKey={activeAction}
                    visible={activeTab === 'breakdown'}
                  />
                )}
              </div>
            </div>
          </>
        )}
      </main>

      <footer className={styles.footer}>
        <Link to="/docs">Documentation Hub</Link>
        &nbsp;·&nbsp; AtomWorldBench &nbsp;·&nbsp;
        Data generated with <code>src/scripts/generate_gh_pages_data.py</code>
      </footer>
    </div>
  )
}
