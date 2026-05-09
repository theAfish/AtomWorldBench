import { useEffect, useRef } from 'react'
import Plotly from 'plotly.js-dist-min'
import type { BenchmarkData } from '../../../types'

const WRONG_TYPES = [
  'StructureMismatch',
  'AtomCountMismatch',
  'CIFParsingError',
  'OutputFormatError',
] as const

const TYPE_COLORS: Record<string, string> = {
  Correct: '#3d9970',
  StructureMismatch: '#ff851b',
  AtomCountMismatch: '#4c78a8',
  CIFParsingError: '#e15759',
  OutputFormatError: '#9467bd',
  Other: '#aec7e8',
}

interface Props {
  data: BenchmarkData
  actionKey: string
  visible: boolean
}

export function BreakdownChart({ data, actionKey, visible }: Props) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!ref.current) return

    const { models, model_labels, action_labels } = data
    const actionLabel = action_labels[actionKey] ?? actionKey
    const modelLabels = models.map(m => model_labels[m] ?? m)

    const correct: (number | null)[] = []
    const wrongByType: Record<string, (number | null)[]> = Object.fromEntries(
      WRONG_TYPES.map(t => [t, []]),
    )
    const other: (number | null)[] = []

    for (const m of models) {
      const d = data.data[m]?.[actionKey]
      const total = d?.total ?? 0
      const sc = d?.success_count ?? 0
      const et = d?.error_types ?? {}

      correct.push(total > 0 ? (sc / total) * 100 : null)

      let knownWrong = 0
      for (const t of WRONG_TYPES) {
        const cnt = et[t] ?? 0
        wrongByType[t].push(total > 0 ? (cnt / total) * 100 : null)
        knownWrong += cnt
      }
      const otherCnt = (d?.error_count ?? 0) - knownWrong
      other.push(total > 0 ? (Math.max(0, otherCnt) / total) * 100 : null)
    }

    const baseTraces = [
      { name: 'Correct', y: correct },
      ...WRONG_TYPES.map(t => ({ name: t, y: wrongByType[t] })),
      { name: 'Other', y: other },
    ]

    const traces = baseTraces.map(t => ({
      ...t,
      type: 'bar',
      x: modelLabels,
      marker: { color: TYPE_COLORS[t.name] ?? '#ccc' },
      text: (t.y as (number | null)[]).map(v =>
        v != null && v >= 8 ? `${v.toFixed(1)}%` : '',
      ),
      textposition: 'inside',
      insidetextanchor: 'middle',
      textfont: { size: 11 },
      hovertemplate: `<b>%{x}</b><br>${t.name}: %{y:.1f}%<extra></extra>`,
      showlegend: true,
    }))

    const layout = {
      barmode: 'stack',
      title: { text: actionLabel, font: { size: 14 } },
      yaxis: { title: 'Proportion (%)', range: [0, 100], ticksuffix: '%' },
      xaxis: { tickangle: -30, automargin: true },
      legend: { orientation: 'h', y: -0.22 },
      margin: { l: 55, r: 20, t: 50, b: 130 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      bargap: 0.22,
    }

    Plotly.newPlot(ref.current, traces, layout, {
      responsive: true,
      displayModeBar: false,
    })

    const el = ref.current
    return () => Plotly.purge(el)
  }, [data, actionKey])

  useEffect(() => {
    if (visible && ref.current) Plotly.Plots.resize(ref.current)
  }, [visible])

  return <div ref={ref} style={{ height: 420 }} />
}
