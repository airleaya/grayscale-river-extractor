import { useEffect, useMemo, useRef, useState } from 'react'

type Point = {
  x: number
  y: number
}

type Size = {
  width: number
  height: number
}

type HoverSample = {
  x: number
  y: number
  rgba: [number, number, number, number]
}

type LoadedImage = {
  width: number
  height: number
  sourceCanvas: HTMLCanvasElement
  sourceContext: CanvasRenderingContext2D
  imageData: Uint8ClampedArray
}

type PixelRasterViewerProps = {
  imageUrl: string
  alt: string
  persistedView: PersistedView | null
  onViewChange: (view: PersistedView) => void
}

type VisibleRegion = {
  sourceX: number
  sourceY: number
  sourceWidth: number
  sourceHeight: number
  drawX: number
  drawY: number
}

export type PersistedView = {
  scale: number
  centerXRatio: number
  centerYRatio: number
}

const MIN_CONTINUOUS_SCALE = 0.1
const DISCRETE_SCALES = [1, 2, 4, 8, 16, 32, 64]
const GRID_SCALE_THRESHOLD = 8
const PAN_MARGIN = 64
const THUMBNAIL_SIZE = 180

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

function getCanvasPoint(canvas: HTMLCanvasElement, event: MouseEvent | WheelEvent): Point {
  const rect = canvas.getBoundingClientRect()
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
  }
}

function formatRgba(sample: HoverSample | null): string {
  if (sample === null) {
    return '--'
  }

  const [r, g, b, a] = sample.rgba
  return `${r}, ${g}, ${b}, ${a}`
}

function loadImage(source: string): Promise<LoadedImage> {
  return new Promise((resolve, reject) => {
    const image = new Image()
    if (source.startsWith('http://') || source.startsWith('https://')) {
      image.crossOrigin = 'anonymous'
    }
    image.onload = () => {
      const sourceCanvas = document.createElement('canvas')
      sourceCanvas.width = image.naturalWidth
      sourceCanvas.height = image.naturalHeight

      const sourceContext = sourceCanvas.getContext('2d')
      if (sourceContext === null) {
        reject(new Error('无法创建图像采样上下文。'))
        return
      }

      sourceContext.imageSmoothingEnabled = false
      sourceContext.drawImage(image, 0, 0)
      let imageData: Uint8ClampedArray
      try {
        imageData = sourceContext.getImageData(0, 0, sourceCanvas.width, sourceCanvas.height).data
      } catch {
        reject(new Error('图像已加载，但浏览器阻止了像素采样。请检查静态资源跨域配置。'))
        return
      }

      resolve({
        width: image.naturalWidth,
        height: image.naturalHeight,
        sourceCanvas,
        sourceContext,
        imageData,
      })
    }
    image.onerror = () => {
      reject(new Error(`无法加载图像: ${source}`))
    }
    image.src = source
  })
}

function chooseInitialScale(fitScale: number): number {
  if (fitScale < 1) {
    return fitScale
  }

  const safeFitScale = Math.min(fitScale, DISCRETE_SCALES[DISCRETE_SCALES.length - 1])
  let chosen = 1

  for (const scale of DISCRETE_SCALES) {
    if (scale <= safeFitScale) {
      chosen = scale
    }
  }

  return chosen
}

function centerOffset(image: LoadedImage, scale: number, viewport: Size): Point {
  return {
    x: (viewport.width - image.width * scale) / 2,
    y: (viewport.height - image.height * scale) / 2,
  }
}

function buildFitView(image: LoadedImage, viewport: Size) {
  const fitScale = Math.min(viewport.width / image.width, viewport.height / image.height)
  const scale = chooseInitialScale(Math.max(fitScale, MIN_CONTINUOUS_SCALE))

  return {
    scale,
    offset: centerOffset(image, scale, viewport),
  }
}

function buildPersistedView(image: LoadedImage, viewport: Size, scale: number, offset: Point): PersistedView {
  const visibleCenterX = clamp((viewport.width / 2 - offset.x) / scale, 0, image.width)
  const visibleCenterY = clamp((viewport.height / 2 - offset.y) / scale, 0, image.height)

  return {
    scale,
    centerXRatio: image.width > 0 ? visibleCenterX / image.width : 0.5,
    centerYRatio: image.height > 0 ? visibleCenterY / image.height : 0.5,
  }
}

function restoreViewFromPersisted(image: LoadedImage, viewport: Size, persistedView: PersistedView) {
  const nextScale = clamp(
    persistedView.scale,
    MIN_CONTINUOUS_SCALE,
    DISCRETE_SCALES[DISCRETE_SCALES.length - 1],
  )
  const centerX = clamp(persistedView.centerXRatio, 0, 1) * image.width
  const centerY = clamp(persistedView.centerYRatio, 0, 1) * image.height
  const nextOffset = clampOffset(
    {
      x: viewport.width / 2 - centerX * nextScale,
      y: viewport.height / 2 - centerY * nextScale,
    },
    image,
    nextScale,
    viewport,
  )

  return {
    scale: nextScale,
    offset: nextOffset,
  }
}

function clampOffset(candidateOffset: Point, image: LoadedImage, scale: number, viewport: Size): Point {
  const scaledWidth = image.width * scale
  const scaledHeight = image.height * scale

  if (scaledWidth <= viewport.width) {
    candidateOffset.x = (viewport.width - scaledWidth) / 2
  } else {
    candidateOffset.x = clamp(candidateOffset.x, viewport.width - scaledWidth - PAN_MARGIN, PAN_MARGIN)
  }

  if (scaledHeight <= viewport.height) {
    candidateOffset.y = (viewport.height - scaledHeight) / 2
  } else {
    candidateOffset.y = clamp(candidateOffset.y, viewport.height - scaledHeight - PAN_MARGIN, PAN_MARGIN)
  }

  return candidateOffset
}

function computeVisibleRegion(image: LoadedImage, offset: Point, scale: number, viewport: Size): VisibleRegion | null {
  const drawLeft = Math.max(offset.x, 0)
  const drawTop = Math.max(offset.y, 0)
  const drawRight = Math.min(offset.x + image.width * scale, viewport.width)
  const drawBottom = Math.min(offset.y + image.height * scale, viewport.height)

  if (drawRight <= drawLeft || drawBottom <= drawTop) {
    return null
  }

  const sourceX = Math.max(0, Math.floor((drawLeft - offset.x) / scale))
  const sourceY = Math.max(0, Math.floor((drawTop - offset.y) / scale))
  const sourceWidth = Math.min(image.width - sourceX, Math.ceil((drawRight - drawLeft) / scale))
  const sourceHeight = Math.min(image.height - sourceY, Math.ceil((drawBottom - drawTop) / scale))

  return {
    sourceX,
    sourceY,
    sourceWidth,
    sourceHeight,
    drawX: offset.x + sourceX * scale,
    drawY: offset.y + sourceY * scale,
  }
}

function getNextScale(currentScale: number, deltaY: number): number {
  if (currentScale < 1) {
    if (deltaY < 0) {
      const zoomed = currentScale * 1.14
      return zoomed >= 1 ? 2 : clamp(zoomed, MIN_CONTINUOUS_SCALE, 1)
    }

    return clamp(currentScale / 1.14, MIN_CONTINUOUS_SCALE, 1)
  }

  const currentIndex = DISCRETE_SCALES.indexOf(currentScale)
  const safeIndex = currentIndex === -1 ? 0 : currentIndex

  if (deltaY < 0) {
    return DISCRETE_SCALES[Math.min(safeIndex + 1, DISCRETE_SCALES.length - 1)]
  }

  if (safeIndex === 0) {
    return 0.9
  }

  return DISCRETE_SCALES[safeIndex - 1]
}

function sampleRgba(image: LoadedImage, x: number, y: number): [number, number, number, number] {
  const offset = (y * image.width + x) * 4
  return [
    image.imageData[offset],
    image.imageData[offset + 1],
    image.imageData[offset + 2],
    image.imageData[offset + 3],
  ]
}

function renderScaledImage(context: CanvasRenderingContext2D, image: LoadedImage, region: VisibleRegion, scale: number) {
  context.imageSmoothingEnabled = true
  context.drawImage(
    image.sourceCanvas,
    region.sourceX,
    region.sourceY,
    region.sourceWidth,
    region.sourceHeight,
    region.drawX,
    region.drawY,
    region.sourceWidth * scale,
    region.sourceHeight * scale,
  )
}

function renderPixelBlocks(context: CanvasRenderingContext2D, image: LoadedImage, region: VisibleRegion, scale: number) {
  for (let y = 0; y < region.sourceHeight; y += 1) {
    for (let x = 0; x < region.sourceWidth; x += 1) {
      const sourceX = region.sourceX + x
      const sourceY = region.sourceY + y
      const [r, g, b, a] = sampleRgba(image, sourceX, sourceY)
      context.fillStyle = `rgba(${r}, ${g}, ${b}, ${a / 255})`
      context.fillRect(region.drawX + x * scale, region.drawY + y * scale, scale, scale)
    }
  }
}

function drawGrid(context: CanvasRenderingContext2D, image: LoadedImage, offset: Point, scale: number, viewport: Size) {
  const startX = Math.max(0, Math.floor((-offset.x) / scale))
  const endX = Math.min(image.width, Math.ceil((viewport.width - offset.x) / scale))
  const startY = Math.max(0, Math.floor((-offset.y) / scale))
  const endY = Math.min(image.height, Math.ceil((viewport.height - offset.y) / scale))

  context.save()
  context.strokeStyle = 'rgba(236, 229, 212, 0.22)'
  context.lineWidth = 1
  context.beginPath()

  for (let x = startX; x <= endX; x += 1) {
    const drawX = Math.round(offset.x + x * scale) + 0.5
    context.moveTo(drawX, Math.max(offset.y, 0))
    context.lineTo(drawX, Math.min(offset.y + image.height * scale, viewport.height))
  }

  for (let y = startY; y <= endY; y += 1) {
    const drawY = Math.round(offset.y + y * scale) + 0.5
    context.moveTo(Math.max(offset.x, 0), drawY)
    context.lineTo(Math.min(offset.x + image.width * scale, viewport.width), drawY)
  }

  context.stroke()
  context.restore()
}

function drawThumbnail(canvas: HTMLCanvasElement, image: LoadedImage, offset: Point, scale: number, viewport: Size) {
  const context = canvas.getContext('2d')
  if (context === null) {
    return
  }

  const fitScale = Math.min(THUMBNAIL_SIZE / image.width, THUMBNAIL_SIZE / image.height)
  const thumbWidth = Math.max(Math.round(image.width * fitScale), 1)
  const thumbHeight = Math.max(Math.round(image.height * fitScale), 1)
  const thumbOffset = {
    x: (THUMBNAIL_SIZE - thumbWidth) / 2,
    y: (THUMBNAIL_SIZE - thumbHeight) / 2,
  }

  canvas.width = THUMBNAIL_SIZE
  canvas.height = THUMBNAIL_SIZE

  context.clearRect(0, 0, THUMBNAIL_SIZE, THUMBNAIL_SIZE)
  context.fillStyle = '#0f1515'
  context.fillRect(0, 0, THUMBNAIL_SIZE, THUMBNAIL_SIZE)
  context.imageSmoothingEnabled = false
  context.drawImage(image.sourceCanvas, thumbOffset.x, thumbOffset.y, thumbWidth, thumbHeight)

  const visibleLeft = clamp((-offset.x) / scale, 0, image.width)
  const visibleTop = clamp((-offset.y) / scale, 0, image.height)
  const visibleWidth = clamp(viewport.width / scale, 0, image.width)
  const visibleHeight = clamp(viewport.height / scale, 0, image.height)

  context.strokeStyle = 'rgba(255, 237, 189, 0.92)'
  context.lineWidth = 2
  context.strokeRect(
    thumbOffset.x + visibleLeft * fitScale,
    thumbOffset.y + visibleTop * fitScale,
    Math.max(visibleWidth * fitScale, 2),
    Math.max(visibleHeight * fitScale, 2),
  )
}

export function PixelRasterViewer({ imageUrl, alt, persistedView, onViewChange }: PixelRasterViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const frameRef = useRef<HTMLDivElement | null>(null)
  const thumbnailRef = useRef<HTMLCanvasElement | null>(null)
  const dragOriginRef = useRef<Point | null>(null)
  const dragOffsetRef = useRef<Point>({ x: 0, y: 0 })
  const hasInitializedViewRef = useRef(false)

  const [loadedImage, setLoadedImage] = useState<LoadedImage | null>(null)
  const [viewport, setViewport] = useState<Size>({ width: 0, height: 0 })
  const [scale, setScale] = useState(1)
  const [offset, setOffset] = useState<Point>({ x: 0, y: 0 })
  const [hoverSample, setHoverSample] = useState<HoverSample | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const scaleMode = useMemo(() => (scale <= 1 ? '连续缩放' : '像素重渲染'), [scale])

  useEffect(() => {
    const frame = frameRef.current
    if (frame === null) {
      return
    }

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (!entry) {
        return
      }

      const nextViewport = {
        width: Math.max(Math.round(entry.contentRect.width), 1),
        height: Math.max(Math.round(entry.contentRect.height), 1),
      }
      setViewport((current) => {
        if (current.width === nextViewport.width && current.height === nextViewport.height) {
          return current
        }
        return nextViewport
      })
    })

    observer.observe(frame)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    let isActive = true
    hasInitializedViewRef.current = false

    setLoadedImage(null)
    setErrorMessage(null)
    setHoverSample(null)

    loadImage(imageUrl)
      .then((image) => {
        if (!isActive) {
          return
        }

        setLoadedImage(image)
      })
      .catch((error: Error) => {
        if (!isActive) {
          return
        }

        setErrorMessage(error.message)
      })

    return () => {
      isActive = false
    }
  }, [imageUrl])

  useEffect(() => {
    if (loadedImage === null || viewport.width === 0 || viewport.height === 0 || hasInitializedViewRef.current) {
      return
    }

    const fitView =
      persistedView === null
        ? buildFitView(loadedImage, viewport)
        : restoreViewFromPersisted(loadedImage, viewport, persistedView)
    hasInitializedViewRef.current = true
    dragOffsetRef.current = fitView.offset
    setScale(fitView.scale)
    setOffset(fitView.offset)
  }, [loadedImage, persistedView, viewport])

  useEffect(() => {
    const canvas = canvasRef.current
    if (canvas === null || loadedImage === null || viewport.width === 0 || viewport.height === 0) {
      return
    }

    const devicePixelRatio = window.devicePixelRatio || 1
    canvas.width = Math.round(viewport.width * devicePixelRatio)
    canvas.height = Math.round(viewport.height * devicePixelRatio)
    canvas.style.width = `${viewport.width}px`
    canvas.style.height = `${viewport.height}px`

    const context = canvas.getContext('2d')
    if (context === null) {
      return
    }

    context.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0)
    context.clearRect(0, 0, viewport.width, viewport.height)
    context.fillStyle = '#090d0e'
    context.fillRect(0, 0, viewport.width, viewport.height)

    const region = computeVisibleRegion(loadedImage, offset, scale, viewport)
    if (region !== null) {
      if (scale <= 1) {
        renderScaledImage(context, loadedImage, region, scale)
      } else {
        renderPixelBlocks(context, loadedImage, region, scale)
      }
    }

    if (scale >= GRID_SCALE_THRESHOLD) {
      drawGrid(context, loadedImage, offset, scale, viewport)
    }
  }, [loadedImage, offset, scale, viewport])

  useEffect(() => {
    if (loadedImage === null || thumbnailRef.current === null || viewport.width === 0 || viewport.height === 0) {
      return
    }

    drawThumbnail(thumbnailRef.current, loadedImage, offset, scale, viewport)
  }, [loadedImage, offset, scale, viewport])

  useEffect(() => {
    const frame = frameRef.current
    if (frame === null) {
      return
    }

    function stopScroll(event: WheelEvent) {
      event.preventDefault()
    }

    frame.addEventListener('wheel', stopScroll, { passive: false })
    return () => {
      frame.removeEventListener('wheel', stopScroll)
    }
  }, [])

  useEffect(() => {
    if (loadedImage === null || viewport.width === 0 || viewport.height === 0 || !hasInitializedViewRef.current) {
      return
    }

    const nextOffset = clampOffset({ ...offset }, loadedImage, scale, viewport)
    dragOffsetRef.current = nextOffset
    setOffset((current) => {
      if (current.x === nextOffset.x && current.y === nextOffset.y) {
        return current
      }
      return nextOffset
    })
  }, [loadedImage, viewport.width, viewport.height])

  useEffect(() => {
    if (loadedImage === null || viewport.width === 0 || viewport.height === 0 || !hasInitializedViewRef.current) {
      return
    }

    onViewChange(buildPersistedView(loadedImage, viewport, scale, offset))
  }, [loadedImage, offset, onViewChange, scale, viewport])

  function handleWheel(event: React.WheelEvent<HTMLCanvasElement>) {
    if (loadedImage === null || canvasRef.current === null || viewport.width === 0 || viewport.height === 0) {
      return
    }

    event.preventDefault()

    const pointer = getCanvasPoint(canvasRef.current, event.nativeEvent)
    const imageSpaceX = (pointer.x - offset.x) / scale
    const imageSpaceY = (pointer.y - offset.y) / scale
    const nextScale = getNextScale(scale, event.deltaY)
    const nextOffset = clampOffset(
      {
        x: pointer.x - imageSpaceX * nextScale,
        y: pointer.y - imageSpaceY * nextScale,
      },
      loadedImage,
      nextScale,
      viewport,
    )

    dragOffsetRef.current = nextOffset
    setScale(nextScale)
    setOffset(nextOffset)
  }

  function handleMouseDown(event: React.MouseEvent<HTMLCanvasElement>) {
    dragOriginRef.current = { x: event.clientX, y: event.clientY }
  }

  function handleMouseMove(event: React.MouseEvent<HTMLCanvasElement>) {
    if (loadedImage === null || canvasRef.current === null || viewport.width === 0 || viewport.height === 0) {
      return
    }

    const pointer = getCanvasPoint(canvasRef.current, event.nativeEvent)
    const pixelX = Math.floor((pointer.x - offset.x) / scale)
    const pixelY = Math.floor((pointer.y - offset.y) / scale)

    if (pixelX >= 0 && pixelY >= 0 && pixelX < loadedImage.width && pixelY < loadedImage.height) {
      setHoverSample({
        x: pixelX,
        y: pixelY,
        rgba: sampleRgba(loadedImage, pixelX, pixelY),
      })
    } else {
      setHoverSample(null)
    }

    if (dragOriginRef.current === null) {
      return
    }

    const deltaX = event.clientX - dragOriginRef.current.x
    const deltaY = event.clientY - dragOriginRef.current.y
    const nextOffset = clampOffset(
      {
        x: dragOffsetRef.current.x + deltaX,
        y: dragOffsetRef.current.y + deltaY,
      },
      loadedImage,
      scale,
      viewport,
    )

    setOffset(nextOffset)
  }

  function handleMouseUp() {
    dragOriginRef.current = null
    dragOffsetRef.current = offset
  }

  function handleMouseLeave() {
    dragOriginRef.current = null
    dragOffsetRef.current = offset
    setHoverSample(null)
  }

  function zoomBy(factor: number) {
    if (loadedImage === null || viewport.width === 0 || viewport.height === 0) {
      return
    }

    const pointer = {
      x: viewport.width / 2,
      y: viewport.height / 2,
    }
    const nextScale =
      scale <= 1
        ? clamp(scale * factor, MIN_CONTINUOUS_SCALE, 1)
        : DISCRETE_SCALES[
            clamp(DISCRETE_SCALES.indexOf(scale) + (factor > 1 ? 1 : -1), 0, DISCRETE_SCALES.length - 1)
          ]
    const imageSpaceX = (pointer.x - offset.x) / scale
    const imageSpaceY = (pointer.y - offset.y) / scale
    const nextOffset = clampOffset(
      {
        x: pointer.x - imageSpaceX * nextScale,
        y: pointer.y - imageSpaceY * nextScale,
      },
      loadedImage,
      nextScale,
      viewport,
    )

    dragOffsetRef.current = nextOffset
    setScale(nextScale)
    setOffset(nextOffset)
  }

  function resetView() {
    if (loadedImage === null || viewport.width === 0 || viewport.height === 0) {
      return
    }

    const fitView = buildFitView(loadedImage, viewport)
    dragOffsetRef.current = fitView.offset
    setScale(fitView.scale)
    setOffset(fitView.offset)
  }

  function handleThumbnailClick(event: React.MouseEvent<HTMLCanvasElement>) {
    if (loadedImage === null || thumbnailRef.current === null || viewport.width === 0 || viewport.height === 0) {
      return
    }

    const point = getCanvasPoint(thumbnailRef.current, event.nativeEvent)
    const fitScale = Math.min(THUMBNAIL_SIZE / loadedImage.width, THUMBNAIL_SIZE / loadedImage.height)
    const thumbWidth = loadedImage.width * fitScale
    const thumbHeight = loadedImage.height * fitScale
    const thumbOffset = {
      x: (THUMBNAIL_SIZE - thumbWidth) / 2,
      y: (THUMBNAIL_SIZE - thumbHeight) / 2,
    }

    const imageX = clamp((point.x - thumbOffset.x) / fitScale, 0, loadedImage.width)
    const imageY = clamp((point.y - thumbOffset.y) / fitScale, 0, loadedImage.height)
    const nextOffset = clampOffset(
      {
        x: viewport.width / 2 - imageX * scale,
        y: viewport.height / 2 - imageY * scale,
      },
      loadedImage,
      scale,
      viewport,
    )

    dragOffsetRef.current = nextOffset
    setOffset(nextOffset)
  }

  return (
    <div className="pixel-viewer">
      <div className="pixel-toolbar">
        <div className="pixel-toolbar-group">
          <button type="button" className="secondary-button pixel-tool-button" onClick={() => zoomBy(1.2)}>
            放大
          </button>
          <button type="button" className="secondary-button pixel-tool-button" onClick={() => zoomBy(0.8)}>
            缩小
          </button>
          <button type="button" className="secondary-button pixel-tool-button" onClick={resetView}>
            复位
          </button>
        </div>

        <div className="pixel-stats">
          <span>模式: {scaleMode}</span>
          <span>缩放: {scale.toFixed(scale < 1 ? 2 : 0)}x</span>
          <span>区域渲染: 开</span>
          <span>视口: {viewport.width > 0 ? `${viewport.width}×${viewport.height}` : '--'}</span>
          <span>网格: {scale >= GRID_SCALE_THRESHOLD ? '开' : '关'}</span>
          <span>像素: {hoverSample ? `(${hoverSample.x}, ${hoverSample.y})` : '--'}</span>
          <span>RGBA: {formatRgba(hoverSample)}</span>
        </div>
      </div>

      <div className="pixel-viewer-layout">
        <div className="pixel-canvas-frame" ref={frameRef}>
          {errorMessage ? (
            <div className="artifact-placeholder">
              <p>{errorMessage}</p>
            </div>
          ) : (
            <canvas
              ref={canvasRef}
              aria-label={alt}
              className="pixel-canvas"
              onWheel={handleWheel}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseLeave}
            />
          )}
        </div>

        <aside className="thumbnail-panel">
          <div className="thumbnail-header">
            <span className="field-label">缩略图导航</span>
            <p className="muted-copy">点击缩略图可快速跳转到目标区域。</p>
          </div>
          <canvas ref={thumbnailRef} className="thumbnail-canvas" onClick={handleThumbnailClick} />
        </aside>
      </div>
    </div>
  )
}
