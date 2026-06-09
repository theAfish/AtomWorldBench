import { useEffect, useRef } from 'react'
import Plotly from 'plotly.js-dist-min'
import type { BenchmarkData } from '../../../types'
import styles from './HeatmapChart.module.css'
import { useIsMobile } from './useIsMobile'

interface Props {
  data: BenchmarkData
  visible: boolean
}

export function MMDHeatmap({ data, visible }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const isMobile = useIsMobile()
  const chartWidth = isMobile ? Math.max(620, data.models.length * 84 + 140) : undefined

  useEffect(() => {
    if (!ref.current) return

    const { models, actions, model_labels, action_labels } = data
    const z: (number | null)[][] = []
    const zText: string[][] = []

    for (const a of actions) {
      const row: (number | null)[] = []
      const rowText: string[] = []
      for (const m of models) {
        const v = data.data[m]?.[a]?.statistics?.max_dist_mean
        if (v != null && !isNaN(v)) {
          row.push(v)
          rowText.push(v.toFixed(2))
        } else {
          row.push(null)
          rowText.push('—')
        }
      }
      z.push(row)
      zText.push(rowText)
    }

    const trace = {
      type: 'heatmap',
      z,
      x: models.map(m => model_labels[m] ?? m),
      y: actions.map(a => action_labels[a] ?? a),
      text: zText,
      texttemplate: '%{text}',
      textfont: { size: isMobile ? 10 : 12 },
      colorscale: [
        [0, '#1b4c8a'],
        [0.2, '#4c78a8'],
        [0.5, '#9ecae1'],
        [0.8, '#fdd49e'],
        [1, '#d7191c'],
      ],
      colorbar: { title: 'Å', thickness: 14 },
      hovertemplate: '<b>%{x}</b><br>%{y}<br>Mean max-dist: %{text} Å<extra></extra>',
    }

    const layout = {
      width: chartWidth,
      margin: { l: isMobile ? 82 : 130, r: isMobile ? 50 : 70, t: 20, b: isMobile ? 78 : 120 },
      xaxis: {
        tickangle: isMobile ? -20 : -35,
        tickfont: { size: isMobile ? 10 : 12 },
        automargin: true,
      },
      yaxis: { tickfont: { size: isMobile ? 10 : 12 }, automargin: true },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
    }

    Plotly.newPlot(ref.current, [trace], layout, {
      responsive: true,
      displayModeBar: false,
    })

    const el = ref.current
    return () => Plotly.purge(el)
  }, [data, isMobile])

  useEffect(() => {
    if (visible && ref.current) Plotly.Plots.resize(ref.current)
  }, [visible])

  return (
    <div className={styles.scrollArea}>
      <div
        ref={ref}
        className={styles.plot}
        style={{ width: chartWidth ? `${chartWidth}px` : '100%', height: 460 }}
      />
    </div>
  )
}
