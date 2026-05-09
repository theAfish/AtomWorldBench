/**
 * Type shim for plotly.js-dist-min.
 * The dist bundle exports the same API as plotly.js but ships without types,
 * so we re-export the relevant surface from @types/plotly.js-compatible shapes.
 */
declare module 'plotly.js-dist-min' {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  type PlotData = Record<string, any>
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  type Layout = Record<string, any>
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  type Config = Record<string, any>

  interface PlotlyHTMLElement extends HTMLElement {
    data: PlotData[]
    layout: Layout
  }

  namespace Plots {
    function resize(root: string | HTMLElement): void
  }

  function newPlot(
    root: string | HTMLElement,
    data: PlotData[],
    layout?: Layout,
    config?: Config,
  ): Promise<PlotlyHTMLElement>

  function purge(root: string | HTMLElement): void
}
