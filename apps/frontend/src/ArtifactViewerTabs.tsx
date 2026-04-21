import { useCallback, useEffect, useMemo, useState } from 'react'
import { PixelRasterViewer, type PersistedView } from './PixelRasterViewer'
import type { ArtifactRecord, ArtifactViewerResult } from './types'

const API_BASE_URL = 'http://127.0.0.1:8000'
const ARTIFACT_LABELS: Record<string, string> = {
  input_preview: '输入图像',
  user_mask: '用户遮罩',
  auto_mask: '自动遮罩',
  terrain_preprocessed: '预处理',
  flow_direction: '流向',
  flow_accumulation: '汇流累积',
  channel_mask: '河道结果',
}
const PREFERRED_ORDER = [
  'input_preview',
  'user_mask',
  'auto_mask',
  'terrain_preprocessed',
  'flow_direction',
  'flow_accumulation',
  'channel_mask',
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

  return `${API_BASE_URL}/${previewPath}`
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
  const [layerOpacityByKey, setLayerOpacityByKey] = useState<Record<string, number>>({})

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
    if (readyArtifacts.length === 0) {
      setSelectedOverlayKeys([])
      return
    }

    setSelectedOverlayKeys((current) => {
      const filtered = current.filter((key) => readyArtifacts.some((artifact) => artifact.key === key))
      if (filtered.length > 0) {
        return filtered
      }

      const defaultKey = readyArtifacts[readyArtifacts.length - 1]?.key ?? readyArtifacts[0]?.key
      return defaultKey ? [defaultKey] : []
    })
  }, [readyArtifacts])

  const activeArtifact =
    artifactList.find((artifact) => artifact.key === activeKey) ?? artifactList[0] ?? null
  const overlayArtifacts = useMemo(
    () => readyArtifacts.filter((artifact) => selectedOverlayKeys.includes(artifact.key)),
    [readyArtifacts, selectedOverlayKeys],
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
                  <strong>输入在底部，最终输出在顶部</strong>
                </div>
                <div className="metric-card">
                  <span className="field-label">当前顶层</span>
                  <strong>{getArtifactLabel(overlayArtifacts[overlayArtifacts.length - 1])}</strong>
                </div>
              </div>

              <div className="overlay-stage">
                <div className="overlay-canvas">
                  {overlayArtifacts.map((artifact) => (
                    <img
                      key={artifact.key}
                      src={toPreviewUrl(artifact.preview_path) ?? ''}
                      alt={getArtifactLabel(artifact)}
                      className="overlay-image-layer"
                      style={{ opacity: layerOpacityByKey[artifact.key] ?? 0.72 }}
                    />
                  ))}
                </div>

                <aside className="overlay-control-panel">
                  <div className="thumbnail-header">
                    <span className="field-label">图层透明度</span>
                    <p className="muted-copy">默认越靠后的阶段越在上层，最终输出始终在最上层。</p>
                  </div>
                  <div className="overlay-layer-list">
                    {overlayArtifacts.map((artifact) => (
                      <div key={`opacity-${artifact.key}`} className="overlay-layer-card">
                        <div className="overlay-layer-header">
                          <strong>{getArtifactLabel(artifact)}</strong>
                          <span>{Math.round((layerOpacityByKey[artifact.key] ?? 0.72) * 100)}%</span>
                        </div>
                        <input
                          type="range"
                          min={0}
                          max={100}
                          step={1}
                          value={Math.round((layerOpacityByKey[artifact.key] ?? 0.72) * 100)}
                          onChange={(event) => handleOpacityChange(artifact.key, Number(event.target.value) / 100)}
                        />
                      </div>
                    ))}
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

            {activeArtifact.key === 'flow_direction' ? <FlowDirectionLegend /> : null}

            <PixelRasterViewer
              imageUrl={toPreviewUrl(activeArtifact.preview_path) ?? ''}
              alt={getArtifactLabel(activeArtifact)}
              persistedView={persistedView}
              onViewChange={handleViewChange}
            />
          </div>
        )}
      </div>
    </article>
  )
}
