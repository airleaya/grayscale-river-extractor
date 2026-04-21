import { useCallback, useEffect, useMemo, useState } from 'react'
import { PixelRasterViewer, type PersistedView } from './PixelRasterViewer'
import type { ArtifactRecord, ArtifactViewerResult } from './types'

const API_BASE_URL = 'http://127.0.0.1:8000'
const ARTIFACT_LABELS: Record<string, string> = {
  input_preview: '输入图像',
  auto_mask: '自动遮罩',
  terrain_preprocessed: '预处理',
  flow_direction: '流向',
  flow_accumulation: '汇流累积',
  channel_mask: '河道结果',
}
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

function sortArtifacts(artifacts: Record<string, ArtifactRecord>): ArtifactRecord[] {
  const preferredOrder = [
    'input_preview',
    'auto_mask',
    'terrain_preprocessed',
    'flow_direction',
    'flow_accumulation',
    'channel_mask',
  ]

  return [...Object.values(artifacts)].sort((left, right) => {
    const leftIndex = preferredOrder.indexOf(left.key)
    const rightIndex = preferredOrder.indexOf(right.key)
    return leftIndex - rightIndex
  })
}

function getArtifactLabel(artifact: ArtifactRecord): string {
  return ARTIFACT_LABELS[artifact.key] ?? artifact.label
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

  const [activeKey, setActiveKey] = useState<string>('input_preview')
  const [persistedView, setPersistedView] = useState<PersistedView | null>(null)

  useEffect(() => {
    if (artifactList.length === 0) {
      return
    }

    const stillExists = artifactList.some((artifact) => artifact.key === activeKey)
    if (stillExists) {
      return
    }

    setActiveKey(artifactList[0].key)
  }, [activeKey, artifactList])

  const activeArtifact =
    artifactList.find((artifact) => artifact.key === activeKey) ?? artifactList[0] ?? null

  const handleViewChange = useCallback((view: PersistedView) => {
    setPersistedView(view)
  }, [])

  return (
    <article className="panel artifact-panel">
      <div className="artifact-panel-header">
        <div className="artifact-title-row">
          <h2>图像查看区</h2>
          <span className="artifact-inline-note">保留视口缩放、区块位置与像素级查看状态</span>
        </div>
        <div className="artifact-chip-row">
          <span className="artifact-chip">标签 {artifactList.length}</span>
          <span className="artifact-chip">目录 {result?.task_directory ?? '待生成'}</span>
        </div>
      </div>

      {artifactList.length > 0 ? (
        <div className="artifact-tab-list" role="tablist" aria-label="中间产物标签">
          {artifactList.map((artifact) => {
            const isActive = artifact.key === activeArtifact?.key
            const stateClass = artifact.status === 'ready' ? 'artifact-tab ready' : 'artifact-tab pending'

            return (
              <button
                key={artifact.key}
                type="button"
                className={`${stateClass}${isActive ? ' active' : ''}`}
                onClick={() => setActiveKey(artifact.key)}
              >
                <span>{getArtifactLabel(artifact)}</span>
                <small>{artifact.status === 'ready' ? '可查看' : '待生成'}</small>
              </button>
            )
          })}
        </div>
      ) : null}

      <div className="artifact-viewer-frame">
        {activeArtifact === null ? (
          <div className="artifact-placeholder">
            <p>请先上传输入文件并启动任务。</p>
          </div>
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
