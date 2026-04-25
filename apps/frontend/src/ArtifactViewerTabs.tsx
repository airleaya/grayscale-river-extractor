import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  PixelRasterViewer,
  type PersistedView,
  type RasterViewerLayer,
  type BlendMode,
} from './PixelRasterViewer'
import type { ArtifactRecord, ArtifactViewerResult } from './types'

const API_BASE_URL = 'http://127.0.0.1:8000'
const ARTIFACT_LABELS: Record<string, string> = {
  input_preview: '输入图像',
  user_mask: '用户遮罩',
  auto_mask: '自动遮罩',
  terrain_preprocessed: '预处理',
  fill_depth: '填洼深度',
  flow_direction: '流向',
  flow_accumulation: '汇流累积',
  channel_mask: '河道结果',
}
const PREFERRED_ORDER = [
  'input_preview',
  'user_mask',
  'auto_mask',
  'terrain_preprocessed',
  'fill_depth',
  'flow_direction',
  'flow_accumulation',
  'channel_mask',
]
const BLEND_MODE_OPTIONS: ReadonlyArray<{
  value: BlendMode
  label: string
  title: string
}> = [
  { value: 'normal', label: '原图', title: '普通叠加' },
  { value: 'lighten', label: '黑穿', title: '等价于 PS 的变亮，黑色基本不影响下层' },
  { value: 'darken', label: '白穿', title: '等价于 PS 的变暗，白色基本不影响下层' },
]
const FLOW_DIRECTION_LEGEND = [
  { label: '上', color: 'rgb(54, 116, 181)' },
  { label: '右上', color: 'rgb(102, 170, 255)' },
  { label: '右', color: 'rgb(70, 172, 124)' },
  { label: '右下', color: 'rgb(145, 206, 108)' },
  { label: '下', color: 'rgb(214, 165, 61)' },
  { label: '左下', color: 'rgb(220, 116, 69)' },
  { label: '左', color: 'rgb(188, 81, 83)' },
  { label: '左上', color: 'rgb(130, 98, 173)' },
  { label: '无流向', color: 'rgb(18, 24, 32)' },
]

function toPreviewUrl(previewPath: string | null): string | null {
  if (previewPath === null) {
    return null
  }

  if (
    previewPath.startsWith('blob:') ||
    previewPath.startsWith('data:') ||
    previewPath.startsWith('http://') ||
    previewPath.startsWith('https://')
  ) {
    return previewPath
  }

  const normalizedPath = previewPath.startsWith('/') ? previewPath : `/${previewPath}`
  return new URL(normalizedPath, `${API_BASE_URL}/`).toString()
}

function getArtifactLabel(artifact: ArtifactRecord): string {
  return ARTIFACT_LABELS[artifact.key] ?? artifact.label
}

function sortArtifacts(artifacts: Record<string, ArtifactRecord>): ArtifactRecord[] {
  return [...Object.values(artifacts)].sort((left, right) => {
    const leftIndex = PREFERRED_ORDER.indexOf(left.key)
    const rightIndex = PREFERRED_ORDER.indexOf(right.key)
    return leftIndex - rightIndex
  })
}

function getDefaultLayerOpacity(key: string): number {
  switch (key) {
    case 'input_preview':
      return 1
    case 'channel_mask':
      return 1
    case 'user_mask':
      return 0.9
    case 'auto_mask':
      return 0.88
    case 'fill_depth':
      return 0.62
    default:
      return 0.72
  }
}

function getDefaultBlendMode(key: string): BlendMode {
  switch (key) {
    case 'user_mask':
    case 'auto_mask':
    case 'fill_depth':
      return 'lighten'
    default:
      return 'normal'
  }
}

function uniqueKeys(keys: string[]): string[] {
  return [...new Set(keys)]
}

function FlowDirectionLegend() {
  return (
    <section className="flow-legend-block" aria-label="流向图图示">
      <div className="flow-legend-header">
        <span className="field-label">流向图示</span>
        <small>颜色与 D8 方向一一对应</small>
      </div>
      <div className="flow-legend-grid">
        {FLOW_DIRECTION_LEGEND.map((item) => (
          <div key={item.label} className="flow-legend-item">
            <span className="flow-legend-swatch" style={{ backgroundColor: item.color }} aria-hidden="true" />
            <span>{item.label}</span>
          </div>
        ))}
      </div>
    </section>
  )
}

function BlendModeSelector({
  value,
  onChange,
}: {
  value: BlendMode
  onChange: (value: BlendMode) => void
}) {
  return (
    <div className="transparency-toggle-group" role="group" aria-label="图层混合模式">
      {BLEND_MODE_OPTIONS.map((option) => (
        <button
          key={option.value}
          type="button"
          className={`transparency-mode-button${value === option.value ? ' active' : ''}`}
          title={option.title}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  )
}

type ArtifactViewerTabsProps = {
  result: ArtifactViewerResult | null
}

export function ArtifactViewerTabs({ result }: ArtifactViewerTabsProps) {
  const artifactList = useMemo(() => {
    if (result === null) {
      return []
    }

    return sortArtifacts(result.artifacts)
  }, [result])
  const readyArtifacts = useMemo(
    () => artifactList.filter((artifact) => artifact.status === 'ready' && artifact.preview_path !== null),
    [artifactList],
  )

  const [activeKey, setActiveKey] = useState<string>('input_preview')
  const [persistedView, setPersistedView] = useState<PersistedView | null>(null)
  const [overlayMode, setOverlayMode] = useState(false)
  const [selectedOverlayKeys, setSelectedOverlayKeys] = useState<string[]>([])
  const [overlayLayerOrder, setOverlayLayerOrder] = useState<string[]>([])
  const [layerOpacityByKey, setLayerOpacityByKey] = useState<Record<string, number>>({})
  const [layerBlendModeByKey, setLayerBlendModeByKey] = useState<Record<string, BlendMode>>({})
  const [draggingLayerKey, setDraggingLayerKey] = useState<string | null>(null)

  useEffect(() => {
    if (artifactList.length === 0) {
      return
    }

    const stillExists = artifactList.some((artifact) => artifact.key === activeKey)
    if (!stillExists) {
      setActiveKey(artifactList[0].key)
    }
  }, [activeKey, artifactList])

  useEffect(() => {
    setLayerOpacityByKey((current) => {
      let changed = false
      const next: Record<string, number> = {}
      for (const artifact of readyArtifacts) {
        next[artifact.key] = current[artifact.key] ?? getDefaultLayerOpacity(artifact.key)
        if (current[artifact.key] === undefined) {
          changed = true
        }
      }
      return changed || Object.keys(current).length !== Object.keys(next).length ? next : current
    })

    setLayerBlendModeByKey((current) => {
      let changed = false
      const next: Record<string, BlendMode> = {}
      for (const artifact of readyArtifacts) {
        next[artifact.key] = current[artifact.key] ?? getDefaultBlendMode(artifact.key)
        if (current[artifact.key] === undefined) {
          changed = true
        }
      }
      return changed || Object.keys(current).length !== Object.keys(next).length ? next : current
    })
  }, [readyArtifacts])

  useEffect(() => {
    if (readyArtifacts.length === 0) {
      setSelectedOverlayKeys([])
      return
    }

    setSelectedOverlayKeys((current) => {
      const filtered = current.filter((key) => readyArtifacts.some((artifact) => artifact.key === key))
      if (filtered.length > 0) {
        return filtered
      }

      const lastReadyKey = readyArtifacts[readyArtifacts.length - 1]?.key ?? readyArtifacts[0]?.key
      const inputKey = readyArtifacts.some((artifact) => artifact.key === 'input_preview') ? 'input_preview' : null
      return uniqueKeys([inputKey, lastReadyKey].filter((key): key is string => key !== null))
    })
  }, [readyArtifacts])

  useEffect(() => {
    const readyKeys = readyArtifacts.map((artifact) => artifact.key)
    setOverlayLayerOrder((current) => {
      const filtered = current.filter((key) => readyKeys.includes(key))
      const missing = readyKeys.filter((key) => !filtered.includes(key))
      const next = [...filtered, ...missing]
      return next.length === current.length && next.every((key, index) => key === current[index]) ? current : next
    })
  }, [readyArtifacts])

  const activeArtifact =
    artifactList.find((artifact) => artifact.key === activeKey) ?? artifactList[0] ?? null
  const readyArtifactsByKey = useMemo(
    () => Object.fromEntries(readyArtifacts.map((artifact) => [artifact.key, artifact])) as Record<string, ArtifactRecord>,
    [readyArtifacts],
  )
  const overlayArtifacts = useMemo(
    () =>
      overlayLayerOrder
        .filter((key) => selectedOverlayKeys.includes(key))
        .map((key) => readyArtifactsByKey[key])
        .filter((artifact): artifact is ArtifactRecord => artifact !== undefined),
    [overlayLayerOrder, readyArtifactsByKey, selectedOverlayKeys],
  )

  const handleViewChange = useCallback((view: PersistedView) => {
    setPersistedView(view)
  }, [])

  function handleToggleOverlayArtifact(key: string) {
    setSelectedOverlayKeys((current) => {
      if (current.includes(key)) {
        return current.filter((item) => item !== key)
      }

      return [...current, key].sort((left, right) => PREFERRED_ORDER.indexOf(left) - PREFERRED_ORDER.indexOf(right))
    })
  }

  function handleOpacityChange(key: string, opacity: number) {
    setLayerOpacityByKey((current) => ({
      ...current,
      [key]: opacity,
    }))
  }

  function handleBlendModeChange(key: string, mode: BlendMode) {
    setLayerBlendModeByKey((current) => ({
      ...current,
      [key]: mode,
    }))
  }

  function moveOverlayLayer(sourceKey: string, targetKey: string) {
    if (sourceKey === targetKey) {
      return
    }

    setOverlayLayerOrder((current) => {
      const sourceIndex = current.indexOf(sourceKey)
      const targetIndex = current.indexOf(targetKey)
      if (sourceIndex === -1 || targetIndex === -1) {
        return current
      }

      const next = [...current]
      const [movedKey] = next.splice(sourceIndex, 1)
      next.splice(targetIndex, 0, movedKey)
      return next
    })
  }

  function shiftOverlayLayer(key: string, direction: -1 | 1) {
    setOverlayLayerOrder((current) => {
      const sourceIndex = current.indexOf(key)
      if (sourceIndex === -1) {
        return current
      }

      const targetIndex = sourceIndex + direction
      if (targetIndex < 0 || targetIndex >= current.length) {
        return current
      }

      const next = [...current]
      const [movedKey] = next.splice(sourceIndex, 1)
      next.splice(targetIndex, 0, movedKey)
      return next
    })
  }

  function buildViewerLayer(artifact: ArtifactRecord): RasterViewerLayer | null {
    const imageUrl = toPreviewUrl(artifact.preview_path)
    if (imageUrl === null) {
      return null
    }

    return {
      id: artifact.key,
      label: getArtifactLabel(artifact),
      imageUrl,
      opacity: layerOpacityByKey[artifact.key] ?? getDefaultLayerOpacity(artifact.key),
      blendMode: layerBlendModeByKey[artifact.key] ?? getDefaultBlendMode(artifact.key),
    }
  }

  const activeViewerLayers = useMemo(() => {
    if (overlayMode) {
      return overlayArtifacts
        .map((artifact) => buildViewerLayer(artifact))
        .filter((layer): layer is RasterViewerLayer => layer !== null)
    }

    if (activeArtifact === null || activeArtifact.status !== 'ready' || activeArtifact.preview_path === null) {
      return []
    }

    const layer = buildViewerLayer(activeArtifact)
    return layer ? [layer] : []
  }, [activeArtifact, layerBlendModeByKey, layerOpacityByKey, overlayArtifacts, overlayMode])

  function renderLayerControlCard(
    artifact: ArtifactRecord,
    stackIndex: number | null = null,
    draggable = false,
  ) {
    const opacity = layerOpacityByKey[artifact.key] ?? getDefaultLayerOpacity(artifact.key)
    const blendMode = layerBlendModeByKey[artifact.key] ?? getDefaultBlendMode(artifact.key)

    return (
      <div
        key={`control-${artifact.key}`}
        className={`overlay-layer-card${draggingLayerKey === artifact.key ? ' dragging' : ''}${draggable ? ' sortable' : ''}`}
        onDragOver={(event) => {
          if (!draggable || draggingLayerKey === null || draggingLayerKey === artifact.key) {
            return
          }
          event.preventDefault()
          event.dataTransfer.dropEffect = 'move'
        }}
        onDragEnter={(event) => {
          if (!draggable || draggingLayerKey === null || draggingLayerKey === artifact.key) {
            return
          }
          event.preventDefault()
          moveOverlayLayer(draggingLayerKey, artifact.key)
        }}
        onDrop={(event) => {
          if (!draggable) {
            return
          }
          event.preventDefault()
          const sourceKey = event.dataTransfer.getData('text/plain')
          if (sourceKey) {
            moveOverlayLayer(sourceKey, artifact.key)
          }
          setDraggingLayerKey(null)
        }}
        onDragEnd={() => setDraggingLayerKey(null)}
      >
        <div className="overlay-layer-header">
          <strong>{getArtifactLabel(artifact)}</strong>
          <div className="overlay-layer-meta">
            <span>
              {stackIndex === null ? `${Math.round(opacity * 100)}%` : `第 ${stackIndex} 层 · ${Math.round(opacity * 100)}%`}
            </span>
            {draggable ? (
              <>
                <button
                  type="button"
                  className="overlay-layer-order-button"
                  onClick={() => shiftOverlayLayer(artifact.key, -1)}
                  title="上移图层"
                >
                  上移
                </button>
                <button
                  type="button"
                  className="overlay-layer-order-button"
                  onClick={() => shiftOverlayLayer(artifact.key, 1)}
                  title="下移图层"
                >
                  下移
                </button>
                <span
                  className="overlay-layer-drag-hint"
                  draggable={true}
                  onDragStart={(event) => {
                    event.dataTransfer.effectAllowed = 'move'
                    event.dataTransfer.setData('text/plain', artifact.key)
                    setDraggingLayerKey(artifact.key)
                  }}
                  onDragEnd={() => setDraggingLayerKey(null)}
                  title="拖动改变图层顺序"
                >
                  拖动排序
                </span>
              </>
            ) : null}
          </div>
        </div>
        <div className="overlay-layer-actions">
          <label className="overlay-control-label">
            <span>透明度</span>
            <input
              type="range"
              min={0}
              max={100}
              step={1}
              value={Math.round(opacity * 100)}
              onChange={(event) => handleOpacityChange(artifact.key, Number(event.target.value) / 100)}
            />
          </label>
          <label className="overlay-control-label">
            <span>混合模式</span>
            <BlendModeSelector
              value={blendMode}
              onChange={(mode) => handleBlendModeChange(artifact.key, mode)}
            />
          </label>
        </div>
      </div>
    )
  }

  return (
    <article className="panel artifact-panel">
      <div className="artifact-panel-header">
        <div className="artifact-title-row">
          <h2>图像查看区</h2>
          <span className="artifact-inline-note">支持单图像素查看与成果叠图对比</span>
        </div>
        <div className="artifact-chip-row">
          <span className="artifact-chip">标签 {artifactList.length}</span>
          <span className="artifact-chip">目录 {result?.task_directory ?? '待生成'}</span>
          <button
            type="button"
            className={`artifact-chip overlay-toggle-chip${overlayMode ? ' active' : ''}`}
            onClick={() => setOverlayMode((current) => !current)}
          >
            {overlayMode ? '叠图查看中' : '多图层查看'}
          </button>
        </div>
      </div>

      {artifactList.length > 0 ? (
        <div className="artifact-tab-list" role="tablist" aria-label="中间产物标签">
          {artifactList.map((artifact) => {
            const isActive = overlayMode
              ? selectedOverlayKeys.includes(artifact.key)
              : artifact.key === activeArtifact?.key
            const stateClass = artifact.status === 'ready' ? 'artifact-tab ready' : 'artifact-tab pending'

            return (
              <button
                key={artifact.key}
                type="button"
                className={`${stateClass}${isActive ? ' active' : ''}`}
                onClick={() =>
                  overlayMode ? handleToggleOverlayArtifact(artifact.key) : setActiveKey(artifact.key)
                }
              >
                <span>{getArtifactLabel(artifact)}</span>
                <small>{artifact.status === 'ready' ? (overlayMode ? '可叠图' : '可查看') : '待生成'}</small>
              </button>
            )
          })}
        </div>
      ) : null}

      <div className="artifact-viewer-frame">
        {activeArtifact === null ? (
          <div className="artifact-placeholder">
            <p>请先新建任务，并为任务选择输入图像。</p>
          </div>
        ) : overlayMode ? (
          overlayArtifacts.length === 0 ? (
            <div className="artifact-placeholder">
              <p>当前没有可叠加的图层。</p>
              <p className="muted-copy">请选择至少一个已生成的阶段产物。</p>
            </div>
          ) : (
            <div className="overlay-view-layout">
              <div className="artifact-preview-meta">
                <div className="metric-card">
                  <span className="field-label">叠图模式</span>
                  <strong>{overlayArtifacts.length} 个图层</strong>
                </div>
                <div className="metric-card">
                  <span className="field-label">堆叠规则</span>
                  <strong>按右侧图层顺序渲染，越靠下越在上层</strong>
                </div>
                <div className="metric-card">
                  <span className="field-label">当前顶层</span>
                  <strong>{getArtifactLabel(overlayArtifacts[overlayArtifacts.length - 1])}</strong>
                </div>
              </div>

              <div className="overlay-stage">
                <div className="artifact-primary-view">
                  <PixelRasterViewer
                    layers={activeViewerLayers}
                    alt="多图层叠图查看"
                    persistedView={persistedView}
                    onViewChange={handleViewChange}
                  />
                </div>

                <aside className="overlay-control-panel">
                  <div className="thumbnail-header">
                    <span className="field-label">图层设置</span>
                    <p className="muted-copy">拖动图层卡片可排序，越靠下的图层越盖在上面；黑穿等价于变亮，白穿等价于变暗。</p>
                  </div>
                  <div className="overlay-layer-list">
                    {overlayArtifacts.map((artifact, index) =>
                      renderLayerControlCard(artifact, index + 1, true),
                    )}
                  </div>
                </aside>
              </div>
            </div>
          )
        ) : activeArtifact.status !== 'ready' || activeArtifact.preview_path === null ? (
          <div className="artifact-placeholder">
            <p>{getArtifactLabel(activeArtifact)} 尚未生成。</p>
            <p className="muted-copy">当前阶段完成后，这里会自动刷新为对应图像。</p>
          </div>
        ) : (
          <div className="artifact-preview-layout">
            <PixelRasterViewer
              layers={activeViewerLayers}
              alt={getArtifactLabel(activeArtifact)}
              persistedView={persistedView}
              onViewChange={handleViewChange}
            />

            <div className="artifact-preview-meta">
              <div className="metric-card">
                <span className="field-label">当前标签</span>
                <strong>{getArtifactLabel(activeArtifact)}</strong>
              </div>
              <div className="metric-card">
                <span className="field-label">图像尺寸</span>
                <strong>
                  {activeArtifact.width && activeArtifact.height
                    ? `${activeArtifact.width} × ${activeArtifact.height}`
                    : '--'}
                </strong>
              </div>
              <div className="metric-card">
                <span className="field-label">预览路径</span>
                <code className="mono-copy">{activeArtifact.preview_path}</code>
              </div>
            </div>

            <div className="artifact-view-controls">
              {renderLayerControlCard(activeArtifact)}
            </div>

            {activeArtifact.key === 'flow_direction' ? <FlowDirectionLegend /> : null}
          </div>
        )}
      </div>
    </article>
  )
}
