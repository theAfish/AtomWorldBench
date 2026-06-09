import { Link, useLocation } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import { TopBar } from '../../components/TopBar'
import '../../styles/global.css'
import 'highlight.js/styles/github-dark.css'
import styles from './DocsPage.module.css'

const NAV = [
  { to: '/docs', label: 'Overview' },
  {
    group: 'Getting Started',
    items: [{ to: '/installation', label: 'Installation' }],
  },
  {
    group: 'Reference',
    items: [
      { to: '/api-usage', label: 'API Usage' },
      { to: '/python-api', label: 'Python API' },
      { to: '/actions', label: 'Actions' },
      { to: '/cli', label: 'CLI' },
      { to: '/agent-mode', label: 'Agent Mode' },
    ],
  },
  {
    group: 'More',
    items: [
      { to: '/data-generation', label: 'Data Generation' },
      { to: '/contributing', label: 'Contributing' },
      { to: '/citation', label: 'Citation' },
    ],
  },
]

interface NavItem { to: string; label: string }
interface NavGroup { group: string; items: NavItem[] }
type NavEntry = NavItem | NavGroup

function isGroup(e: NavEntry): e is NavGroup {
  return 'group' in e
}

interface Props {
  content: string
  title: string
}

export function DocsPage({ content }: Props) {
  const { pathname } = useLocation()

  /** Recognise SPA-internal paths so we can navigate with React Router
   *  instead of a full-page load (which would 404 on GitHub Pages). */
  const isInternal = (href: string) =>
    href.startsWith('/') && !href.startsWith('//')

  return (
    <div className={styles.page}>
      <TopBar />
      <div className={styles.layout}>
        {/* ── Sidebar ──────────────────────────────────────────────── */}
        <nav className={styles.sidebar} aria-label="Documentation">
          {NAV.map(entry =>
            isGroup(entry) ? (
              <div key={entry.group} className={styles.navGroup}>
                <span className={styles.navGroupLabel}>{entry.group}</span>
                {entry.items.map(({ to, label }) => (
                  <Link
                    key={to}
                    to={to}
                    className={`${styles.navLink} ${pathname === to ? styles.navLinkActive : ''}`}
                  >
                    {label}
                  </Link>
                ))}
              </div>
            ) : (
              <Link
                key={entry.to}
                to={entry.to}
                className={`${styles.navLink} ${pathname === entry.to ? styles.navLinkActive : ''}`}
              >
                {entry.label}
              </Link>
            ),
          )}
        </nav>

        {/* ── Content ──────────────────────────────────────────────── */}
        <article className={styles.content}>
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypeHighlight]}
            components={{
              a({ href, children, node, ...props }) {
                if (href && isInternal(href)) {
                  return (
                    <Link to={href} {...props}>
                      {children}
                    </Link>
                  )
                }
                return <a href={href} target={href ? '_blank' : undefined} rel="noreferrer" {...props}>{children}</a>
              },
              // Admonition-like blockquote handling for !!! tip / !!! info
              // (react-markdown doesn't parse MkDocs admonitions natively, so
              // we convert them with a pre-processing step in the content string)
              // Only apply inline styling when NOT inside a fenced code block
              code({ className, children, node, ...props }) {
                const isBlock = node?.position?.start?.line !== node?.position?.end?.line ||
                  String(children).includes('\n')
                if (isBlock || className?.startsWith('language-')) {
                  return <code className={className} {...props}>{children}</code>
                }
                return (
                  <code className={styles.inlineCode} {...props}>
                    {children}
                  </code>
                )
              },
            }}
          >
            {preprocessMarkdown(content)}
          </ReactMarkdown>
        </article>
      </div>
    </div>
  )
}

/**
 * Convert the two MkDocs-specific constructs used in this project into
 * standard Markdown so react-markdown can render them:
 *
 *   !!! tip "Title"       →  > **💡 Tip — Title**\n> body
 *   !!! info "Title"      →  > **ℹ️ Info — Title**\n> body
 */
function preprocessMarkdown(src: string): string {
  const ICONS: Record<string, string> = {
    tip: '💡 Tip',
    info: 'ℹ️ Info',
    note: '📝 Note',
    warning: '⚠️ Warning',
    danger: '🚨 Danger',
  }

  const lines = src.split('\n')
  const out: string[] = []
  let i = 0

  while (i < lines.length) {
    const line = lines[i]
    // Match:  !!! type "optional title"
    const m = line.match(/^!!!\s+(\w+)(?:\s+"([^"]*)")?/)
    if (m) {
      const type = m[1].toLowerCase()
      const title = m[2] ?? ''
      const icon = ICONS[type] ?? type
      const header = title ? `${icon} — ${title}` : icon
      out.push(`> **${header}**`)
      i++
      // Collect indented body lines
      while (i < lines.length && (lines[i].startsWith('    ') || lines[i] === '')) {
        const bodyLine = lines[i].startsWith('    ') ? lines[i].slice(4) : ''
        out.push(`> ${bodyLine}`)
        i++
      }
    } else {
      out.push(line)
      i++
    }
  }

  return out.join('\n')
}
