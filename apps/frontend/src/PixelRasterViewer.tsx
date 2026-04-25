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

type PreparedLayerImage = {
  id: string
  label: string
  opacity: number
  blendMode: BlendMode
  image: LoadedImage
}

export type BlendMode = 'normal' | 'lighten' | 'darken'

export type RasterViewerLayer = {
  id: string
  label: string
  imageUrl: string
  opacity?: number
  blendMode?: BlendMode
}

type PixelRasterViewerProps = {
  layers: RasterViewerLayer[]
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

async function resolveImageSource(source: string): Promise<{ resolvedSource: string; cleanup?: () => void }> {
  if (
    source.startsWith('blob:') ||
    source.startsWith('data:')
  ) {
    return { resolvedSource: source }
  }

  if (source.startsWith('http://') || source.startsWith('https://')) {
    const response = await fetch(source)
    if (!response.ok) {
      throw new Error(`无法加载图像: ${source} (${response.status})`)
    }

    const blob = await response.blob()
    const objectUrl = URL.createObjectURL(blob)
    return {
      resolvedSource: objectUrl,
      cleanup: () => URL.revokeObjectURL(objectUrl),
    }
  }

  return { resolvedSource: source }
}

function loadImage(source: string): Promise<LoadedImage> {
  return new Promise((resolve, reject) => {
    void (async () => {
      let cleanup: (() => void) | undefined

      try {
        const resolved = await resolveImageSource(source)
        cleanup = resolved.cleanup

        const image = new Image()
        image.decoding = 'async'
        image.onload = () => {
          const sourceCanvas = document.createElement('canvas')
          sourceCanvas.width = image.naturalWidth
          sourceCanvas.height = image.naturalHeight

          const sourceContext = sourceCanvas.getContext('2d')
          if (sourceContext === null) {
            cleanup?.()
            reject(new Error('无法创建图像采样上下文。'))
            return
          }

          sourceContext.imageSmoothingEnabled = false
          sourceContext.drawImage(image, 0, 0)

          let imageData: Uint8ClampedArray
          try {
            imageData = sourceContext.getImageData(0, 0, sourceCanvas.width, sourceCanvas.height).data
          } catch {
            cleanup?.()
            reject(new Error('图像已加载，但浏览器阻止了像素采样。请检查静态资源跨域配置。'))
            return
          }

          cleanup?.()
          resolve({
            width: image.naturalWidth,
            height: image.naturalHeight,
            sourceCanvas,
            sourceContext,
            imageData,
          })
        }

        image.onerror = () => {
          cleanup?.()
          reject(new Error(`无法加载图像: ${source}`))
        }

        image.src = resolved.resolvedSource
      } catch (error) {
        cleanup?.()
        reject(error instanceof Error ? error : new Error(`无法加载图像: ${source}`))
      }
    })()
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

function sampleRgbaSafe(image: LoadedImage, x: number, y: number): [number, number, number, number] {
  if (x < 0 || y < 0 || x >= image.width || y >= image.height) {
    return [0, 0, 0, 0]
  }

  return sampleRgba(image, x, y)
}

function renderScaledImage(
  context: CanvasRenderingContext2D,
  image: LoadedImage,
  region: VisibleRegion,
  scale: number,
  opacity: number,
  blendMode: BlendMode,
) {
  context.save()
  context.globalAlpha = opacity
  context.globalCompositeOperation =
    blendMode === 'lighten' ? 'lighten' : blendMode === 'darken' ? 'darken' : 'source-over'
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
  context.restore()
}

function blendChannel(backdrop: number, source: number, blendMode: BlendMode): number {
  if (blendMode === 'lighten') {
    return Math.max(backdrop, source)
  }

  if (blendMode === 'darken') {
    return Math.min(backdrop, source)
  }

  return source
}

function compositeLayersAtPixel(layers: PreparedLayerImage[], x: number, y: number): [number, number, number, number] {
  let red = 0
  let green = 0
  let blue = 0
  let alpha = 0

  for (const layer of layers) {
    const [sourceRed, sourceGreen, sourceBlue, sourceAlphaByte] = sampleRgbaSafe(layer.image, x, y)
    const sourceAlpha = (sourceAlphaByte / 255) * layer.opacity
    if (sourceAlpha <= 0) {
      continue
    }

    const compositeRed = (1 - alpha) * sourceRed + alpha * blendChannel(red, sourceRed, layer.blendMode)
    const compositeGreen = (1 - alpha) * sourceGreen + alpha * blendChannel(green, sourceGreen, layer.blendMode)
    const compositeBlue = (1 - alpha) * sourceBlue + alpha * blendChannel(blue, sourceBlue, layer.blendMode)
    const nextAlpha = sourceAlpha + alpha * (1 - sourceAlpha)
    const nextPremultipliedRed = sourceAlpha * compositeRed + alpha * (1 - sourceAlpha) * red
    const nextPremultipliedGreen = sourceAlpha * compositeGreen + alpha * (1 - sourceAlpha) * green
    const nextPremultipliedBlue = sourceAlpha * compositeBlue + alpha * (1 - sourceAlpha) * blue

    red = nextAlpha > 0 ? nextPremultipliedRed / nextAlpha : 0
    green = nextAlpha > 0 ? nextPremultipliedGreen / nextAlpha : 0
    blue = nextAlpha > 0 ? nextPremultipliedBlue / nextAlpha : 0
    alpha = nextAlpha
  }

  return [
    Math.round(clamp(red, 0, 255)),
    Math.round(clamp(green, 0, 255)),
    Math.round(clamp(blue, 0, 255)),
    Math.round(clamp(alpha * 255, 0, 255)),
  ]
}

function renderPixelBlocks(
  context: CanvasRenderingContext2D,
  layers: PreparedLayerImage[],
  region: VisibleRegion,
  scale: number,
) {
  for (let y = 0; y < region.sourceHeight; y += 1) {
    for (let x = 0; x < region.sourceWidth; x += 1) {
      const sourceX = region.sourceX + x
      const sourceY = region.sourceY + y
      const [r, g, b, a] = compositeLayersAtPixel(layers, sourceX, sourceY)
      if (a <= 0) {
        continue
      }

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

function drawThumbnail(
  canvas: HTMLCanvasElement,
  layers: PreparedLayerImage[],
  primaryImage: LoadedImage,
  offset: Point,
  scale: number,
  viewport: Size,
) {
  const context = canvas.getContext('2d')
  if (context === null) {
    return
  }

  const fitScale = Math.min(THUMBNAIL_SIZE / primaryImage.width, THUMBNAIL_SIZE / primaryImage.height)
  const thumbWidth = Math.max(Math.round(primaryImage.width * fitScale), 1)
  const thumbHeight = Math.max(Math.round(primaryImage.height * fitScale), 1)
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

  for (const layer of layers) {
    context.save()
    context.globalAlpha = layer.opacity
    context.drawImage(layer.image.sourceCanvas, thumbOffset.x, thumbOffset.y, thumbWidth, thumbHeight)
    context.restore()
  }

  const visibleLeft = clamp((-offset.x) / scale, 0, primaryImage.width)
  const visibleTop = clamp((-offset.y) / scale, 0, primaryImage.height)
  const visibleWidth = clamp(viewport.width / scale, 0, primaryImage.width)
  const visibleHeight = clamp(viewport.height / scale, 0, primaryImage.height)

  context.strokeStyle = 'rgba(255, 237, 189, 0.92)'
  context.lineWidth = 2
  context.strokeRect(
    thumbOffset.x + visibleLeft * fitScale,
    thumbOffset.y + visibleTop * fitScale,
    Math.max(visibleWidth * fitScale, 2),
    Math.max(visibleHeight * fitScale, 2),
  )
}

export function PixelRasterViewer({ layers, alt, persistedView, onViewChange }: PixelRasterViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const frameRef = useRef<HTMLDivElement | null>(null)
  const thumbnailRef = useRef<HTMLCanvasElement | null>(null)
  const dragOriginRef = useRef<Point | null>(null)
  const dragOffsetRef = useRef<Point>({ x: 0, y: 0 })
  const hasInitializedViewRef = useRef(false)

  const [rawLayerImagesById, setRawLayerImagesById] = useState<Record<string, LoadedImage>>({})
  const [viewport, setViewport] = useState<Size>({ width: 0, height: 0 })
  const [scale, setScale] = useState(1)
  const [offset, setOffset] = useState<Point>({ x: 0, y: 0 })
  const [hoverSample, setHoverSample] = useState<HoverSample | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  const scaleMode = useMemo(() => (scale <= 1 ? '连续缩放' : '像素重渲染'), [scale])
  const loadSignature = useMemo(
    () => layers.map((layer) => `${layer.id}:${layer.imageUrl}`).join('|'),
    [layers],
  )

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
    setHoverSample(null)
    setErrorMessage(null)

    if (layers.length === 0) {
      setRawLayerImagesById({})
      setIsLoading(false)
      return () => {
        isActive = false
      }
    }

    setIsLoading(true)

    void Promise.all(
      layers.map(async (layer) => ({
        id: layer.id,
        image: await loadImage(layer.imageUrl),
      })),
    )
      .then((loadedLayers) => {
        if (!isActive) {
          return
        }

        const nextLayerMap = Object.fromEntries(
          loadedLayers.map((layer) => [layer.id, layer.image]),
        ) as Record<string, LoadedImage>
        setRawLayerImagesById(nextLayerMap)
        setIsLoading(false)
      })
      .catch((error: Error) => {
        if (!isActive) {
          return
        }

        setRawLayerImagesById({})
        setErrorMessage(error.message)
        setIsLoading(false)
      })

    return () => {
      isActive = false
    }
  }, [loadSignature])

  const preparedLayers = useMemo(
    () =>
      layers
        .map((layer) => {
          const rawImage = rawLayerImagesById[layer.id]
          if (!rawImage) {
            return null
          }

          return {
            id: layer.id,
            label: layer.label,
            opacity: clamp(layer.opacity ?? 1, 0, 1),
            blendMode: layer.blendMode ?? 'normal',
            image: rawImage,
          } satisfies PreparedLayerImage
        })
        .filter((layer): layer is PreparedLayerImage => layer !== null),
    [layers, rawLayerImagesById],
  )

  const primaryLayer = preparedLayers[0] ?? null
  const primaryImage = primaryLayer?.image ?? null

  useEffect(() => {
    if (primaryImage === null || viewport.width === 0 || viewport.height === 0 || hasInitializedViewRef.current) {
      return
    }

    const fitView =
      persistedView === null
        ? buildFitView(primaryImage, viewport)
        : restoreViewFromPersisted(primaryImage, viewport, persistedView)
    hasInitializedViewRef.current = true
    dragOffsetRef.current = fitView.offset
    setScale(fitView.scale)
    setOffset(fitView.offset)
  }, [primaryImage, persistedView, viewport])

  useEffect(() => {
    const canvas = canvasRef.current
    if (
      canvas === null ||
      primaryImage === null ||
      preparedLayers.length === 0 ||
      viewport.width === 0 ||
      viewport.height === 0
    ) {
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

    const region = computeVisibleRegion(primaryImage, offset, scale, viewport)
    if (region !== null) {
      if (scale <= 1) {
        for (const layer of preparedLayers) {
          const layerRegion = computeVisibleRegion(layer.image, offset, scale, viewport)
          if (layerRegion === null || layer.opacity <= 0) {
            continue
          }

          renderScaledImage(context, layer.image, layerRegion, scale, layer.opacity, layer.blendMode)
        }
      } else {
        renderPixelBlocks(context, preparedLayers, region, scale)
      }
    }

    if (scale >= GRID_SCALE_THRESHOLD) {
      drawGrid(context, primaryImage, offset, scale, viewport)
    }
  }, [offset, preparedLayers, primaryImage, scale, viewport])

  useEffect(() => {
    if (
      primaryImage === null ||
      preparedLayers.length === 0 ||
      thumbnailRef.current === null ||
      viewport.width === 0 ||
      viewport.height === 0
    ) {
      return
    }

    drawThumbnail(thumbnailRef.current, preparedLayers, primaryImage, offset, scale, viewport)
  }, [offset, preparedLayers, primaryImage, scale, viewport])

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
    if (primaryImage === null || viewport.width === 0 || viewport.height === 0 || !hasInitializedViewRef.current) {
      return
    }

    const nextOffset = clampOffset({ ...offset }, primaryImage, scale, viewport)
    dragOffsetRef.current = nextOffset
    setOffset((current) => {
      if (current.x === nextOffset.x && current.y === nextOffset.y) {
        return current
      }
      return nextOffset
    })
  }, [primaryImage, scale, viewport.height, viewport.width])

  useEffect(() => {
    if (primaryImage === null || viewport.width === 0 || viewport.height === 0 || !hasInitializedViewRef.current) {
      return
    }

    onViewChange(buildPersistedView(primaryImage, viewport, scale, offset))
  }, [offset, onViewChange, primaryImage, scale, viewport])

  function handleWheel(event: React.WheelEvent<HTMLCanvasElement>) {
    if (primaryImage === null || canvasRef.current === null || viewport.width === 0 || viewport.height === 0) {
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
      primaryImage,
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
    if (
      primaryImage === null ||
      canvasRef.current === null ||
      viewport.width === 0 ||
      viewport.height === 0 ||
      preparedLayers.length === 0
    ) {
      return
    }

    const pointer = getCanvasPoint(canvasRef.current, event.nativeEvent)
    const pixelX = Math.floor((pointer.x - offset.x) / scale)
    const pixelY = Math.floor((pointer.y - offset.y) / scale)

    if (pixelX >= 0 && pixelY >= 0 && pixelX < primaryImage.width && pixelY < primaryImage.height) {
      setHoverSample({
        x: pixelX,
        y: pixelY,
        rgba: compositeLayersAtPixel(preparedLayers, pixelX, pixelY),
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
      primaryImage,
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
    if (primaryImage === null || viewport.width === 0 || viewport.height === 0) {
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
      primaryImage,
      nextScale,
      viewport,
    )

    dragOffsetRef.current = nextOffset
    setScale(nextScale)
    setOffset(nextOffset)
  }

  function resetView() {
    if (primaryImage === null || viewport.width === 0 || viewport.height === 0) {
      return
    }

    const fitView = buildFitView(primaryImage, viewport)
    dragOffsetRef.current = fitView.offset
    setScale(fitView.scale)
    setOffset(fitView.offset)
  }

  function handleThumbnailClick(event: React.MouseEvent<HTMLCanvasElement>) {
    if (primaryImage === null || thumbnailRef.current === null || viewport.width === 0 || viewport.height === 0) {
      return
    }

    const point = getCanvasPoint(thumbnailRef.current, event.nativeEvent)
    const fitScale = Math.min(THUMBNAIL_SIZE / primaryImage.width, THUMBNAIL_SIZE / primaryImage.height)
    const thumbWidth = primaryImage.width * fitScale
    const thumbHeight = primaryImage.height * fitScale
    const thumbOffset = {
      x: (THUMBNAIL_SIZE - thumbWidth) / 2,
      y: (THUMBNAIL_SIZE - thumbHeight) / 2,
    }

    const imageX = clamp((point.x - thumbOffset.x) / fitScale, 0, primaryImage.width)
    const imageY = clamp((point.y - thumbOffset.y) / fitScale, 0, primaryImage.height)
    const nextOffset = clampOffset(
      {
        x: viewport.width / 2 - imageX * scale,
        y: viewport.height / 2 - imageY * scale,
      },
      primaryImage,
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
          <span>图层: {preparedLayers.length}</span>
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
          ) : isLoading ? (
            <div className="artifact-placeholder">
              <p>图像加载中…</p>
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
