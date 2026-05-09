export interface ActionStats {
  total: number
  success_count: number
  error_count: number
  success_rate: number
  error_types: Record<string, number>
  statistics?: {
    max_dist_mean?: number | null
    [key: string]: number | null | undefined
  }
}

export interface BenchmarkData {
  models: string[]
  actions: string[]
  model_labels: Record<string, string>
  action_labels: Record<string, string>
  data: Record<string, Record<string, ActionStats>>
}
