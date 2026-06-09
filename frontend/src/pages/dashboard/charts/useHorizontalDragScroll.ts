import { useEffect, useRef } from 'react'

export function useHorizontalDragScroll<T extends HTMLElement>() {
  const ref = useRef<T>(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return

    let isDragging = false
    let startX = 0
    let startScrollLeft = 0
    let activePointerId: number | null = null

    const onPointerDown = (event: PointerEvent) => {
      if (event.pointerType !== 'mouse' || event.button !== 0) return

      isDragging = true
      activePointerId = event.pointerId
      startX = event.clientX
      startScrollLeft = el.scrollLeft
      el.setPointerCapture(event.pointerId)
      event.preventDefault()
      event.stopPropagation()
    }

    const onPointerMove = (event: PointerEvent) => {
      if (!isDragging || event.pointerId !== activePointerId) return

      el.scrollLeft = startScrollLeft - (event.clientX - startX)
      event.preventDefault()
      event.stopPropagation()
    }

    const stopDragging = (event: PointerEvent) => {
      if (!isDragging || event.pointerId !== activePointerId) return

      isDragging = false
      activePointerId = null
      if (el.hasPointerCapture(event.pointerId)) {
        el.releasePointerCapture(event.pointerId)
      }
      event.preventDefault()
      event.stopPropagation()
    }

    el.addEventListener('pointerdown', onPointerDown, true)
    el.addEventListener('pointermove', onPointerMove, true)
    el.addEventListener('pointerup', stopDragging, true)
    el.addEventListener('pointercancel', stopDragging, true)

    return () => {
      el.removeEventListener('pointerdown', onPointerDown, true)
      el.removeEventListener('pointermove', onPointerMove, true)
      el.removeEventListener('pointerup', stopDragging, true)
      el.removeEventListener('pointercancel', stopDragging, true)
    }
  }, [])

  return ref
}
