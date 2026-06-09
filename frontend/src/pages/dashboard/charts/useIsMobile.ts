import { useEffect, useState } from 'react'

const MOBILE_QUERY = '(max-width: 700px)'

export function useIsMobile() {
  const [isMobile, setIsMobile] = useState(() => {
    if (typeof window === 'undefined') return false
    return window.matchMedia(MOBILE_QUERY).matches
  })

  useEffect(() => {
    const mediaQuery = window.matchMedia(MOBILE_QUERY)

    const update = () => {
      setIsMobile(mediaQuery.matches)
    }

    update()

    if (mediaQuery.addEventListener) {
      mediaQuery.addEventListener('change', update)
      return () => mediaQuery.removeEventListener('change', update)
    }

    mediaQuery.addListener(update)
    return () => mediaQuery.removeListener(update)
  }, [])

  return isMobile
}
