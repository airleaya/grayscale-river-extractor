import { useEffect, useMemo, useRef, useState } from 'react'
import { cancelTask, createTask, getTask, listUploadedFiles, uploadInputFile } from './api'
import { memo } from 'react'
import './App.css'
import './App.cockpit.css'
import { ArtifactViewerTabs } from './ArtifactViewerTabs'
import { DEFAULT_PIPELINE_CONFIG } from './constants'
import type {
  ArtifactViewerResult,
  CreateTaskRequest,
  PipelineConfig,
  TaskStatus,
  TaskSnapshot,
  UploadedFileInfo,
} from './types'

const TASK_POLL_INTERVAL_MS = 1600

function buildOutputPath(fileName: string): string {
  const trimmed = fileName.trim()
  const safeName = trimmed.length > 0 ? trimmed : 'river-output.png'
  return `data/output/${safeName}`
}

function formatFileSize(sizeBytes: number): string {
  if (sizeBytes < 1024) {
    return `${sizeBytes} B`
  }

  if (sizeBytes < 1024 * 1024) {
    return `${(sizeBytes / 1024).toFixed(1)} KB`
  }

  return `${(sizeBytes / (1024 * 1024)).toFixed(2)} MB`
}

function formatRelativeUpdateTime(isoTimestamp: string | null): string {
  if (isoTimestamp === null) {
    return '--'
  }

  const timestamp = Date.parse(isoTimestamp)
  if (Number.isNaN(timestamp)) {
    return '--'
  }

  const deltaSeconds = Math.max(0, Math.round((Date.now() - timestamp) / 1000))
  if (deltaSeconds < 60) {
    return `${deltaSeconds}s 前`
  }

  if (deltaSeconds < 3600) {
    return `${Math.floor(deltaSeconds / 60)}m 前`
  }

  return `${Math.floor(deltaSeconds / 3600)}h 前`
}

function parseTimestampMs(isoTimestamp: string | null): number | null {
  if (isoTimestamp === null) {
    return null
  }

  const timestampMs = Date.parse(isoTimestamp)
  return Number.isNaN(timestampMs) ? null : timestampMs
}

function formatHeartbeatState(status: TaskStatus | null, timestampMs: number | null, nowMs: number): string {
  if (status === null || !['queued', 'running'].includes(status)) {
    return '空闲'
  }

  if (status === 'queued') {
    if (timestampMs === null) {
      return '排队中'
    }

    const deltaSeconds = Math.max(0, Math.round((nowMs - timestampMs) / 1000))
    if (deltaSeconds <= 10) {
      return '排队中'
    }

    return '排队等待'
  }

  if (timestampMs === null) {
    return '未知'
  }

  const deltaSeconds = Math.max(0, Math.round((nowMs - timestampMs) / 1000))
  if (deltaSeconds <= 5) {
    return '活跃'
  }

  if (deltaSeconds <= 15) {
    return '等待心跳'
  }

  return '可能卡住'
}

const STEP_ENTRIES = [
  { key: 'preprocess', title: '预处理', description: '高低映射、平滑、无出口平坡/洼地修复、自动遮罩与用户遮罩。' },
  { key: 'flow_direction', title: '流向', description: 'D8 单流向与有出口平坡导流，后续扩更多模型。' },
  { key: 'flow_accumulation', title: '汇流', description: '累积统计与预览增强。' },
  { key: 'channel_extract', title: '河道', description: '阈值提取与结果输出。' },
  { key: 'postprocess', title: '后处理', description: '连通域、骨架化等入口预留。' },
]

const HELP_TEXT = {
  input: '选择当前处理的主图输入。选择本地文件后会立即在查看器里预览。',
  historyInput: '复用历史上传过的主图素材，避免重复上传。',
  mask: '用户提供的约束图层，非零区域表示允许处理的区域。它只在“用户遮罩”开关开启时参与计算。',
  historyMask: '复用历史上传过的用户遮罩素材，选中后会自动启用“用户遮罩”。',
  output: '设置最终结果文件名，结果会写入 data/output 目录。',
  autoMask: '自动从高程图中提取有效计算区域并生成自动遮罩。它独立于“用户遮罩”开关，开启后会直接参与后续计算。',
  autoMaskBorder: '边界敏感度越高，越容易把边缘背景识别成无效区域。',
  autoMaskTexture: '纹理敏感度越高，越容易把低梯度、低方差的大块区域识别为背景。',
  autoMaskRegion: '自动遮罩保留的最小有效区域面积，小于该面积的孤立小块会被移除。',
  manualMaskEdit: '手动涂抹修正遮罩的入口预留，后续会接入画笔编辑。',
  mapping: '决定灰度值和地形高低的映射关系：亮高或暗高。',
  smooth: '轻度平滑输入高程，减少单像素噪声。',
  fillSinks: '修复封闭洼地和无出口平坡，减少断流。',
  nodata: '指定无效像素值，这些像素会从处理中排除。',
  maskToggle: '启用后把用户提供的遮罩作为额外约束，只在用户遮罩允许的区域内执行算法。',
  smoothKernel: '平滑核越大，地形越平滑，但细节也更容易被抹平。',
  slopeWeight: '严格下降时对最陡方向的偏好强度。',
  rustKernel: '启用后优先使用 Rust 严格 D8 内核；若当前环境未编译扩展，会自动回退到 Python。',
  flatEscapeWeight: '平坡导流时，偏向更快离开平坡区域。',
  outletProximityWeight: '平坡导流时，偏向更靠近强出口的方向。',
  continuityWeight: '平坡导流时，偏向顺着整体出口方向前进，减少机械拐折。',
  outletLengthWeight: '出口段越长时权重越高，影响平坡出口覆盖范围。',
  outletDistanceWeight: '控制平坡内部按距离归属出口段的强度。',
  accumulationPreview: '开启后对汇流预览做对数增强，便于看清主干和支流。',
  threshold: '汇流累积阈值。越低河道越密，越高越偏主干。',
  totalTiles: '任务进度分片数。越大反馈越细，但管理开销也更高。',
  keepArtifacts: '保留中间产物，便于逐阶段查看和调试。',
  uploadMain: '上传当前主图到后端输入目录。',
  uploadMask: '上传当前用户遮罩文件到后端输入目录。',
  startTask: '按当前参数启动任务。',
  cancelTask: '取消当前仍在运行中的任务。',
} as const

const ARTIFACT_KEYS = [
  'input_preview',
  'auto_mask',
  'terrain_preprocessed',
  'flow_direction',
  'flow_accumulation',
  'channel_mask',
] as const

function hasViewerResultMeaningfulChange(
  current: ArtifactViewerResult | null,
  next: ArtifactViewerResult | null,
): boolean {
  if (current === next) {
    return false
  }

  if (current === null || next === null) {
    return current !== next
  }

  if (
    current.task_directory !== next.task_directory ||
    current.metadata_path !== next.metadata_path ||
    current.input_preview !== next.input_preview ||
    current.auto_mask !== next.auto_mask ||
    current.terrain_preprocessed !== next.terrain_preprocessed ||
    current.flow_direction !== next.flow_direction ||
    current.flow_accumulation !== next.flow_accumulation ||
    current.channel_mask !== next.channel_mask
  ) {
    return true
  }

  for (const key of ARTIFACT_KEYS) {
    const left = current.artifacts[key]
    const right = next.artifacts[key]
    if (!left || !right) {
      if (left !== right) {
        return true
      }
      continue
    }

    if (
      left.status !== right.status ||
      left.path !== right.path ||
      left.preview_path !== right.preview_path ||
      left.width !== right.width ||
      left.height !== right.height
    ) {
      return true
    }
  }

  return false
}

const MemoArtifactViewerTabs = memo(ArtifactViewerTabs, (previousProps, nextProps) => {
  return !hasViewerResultMeaningfulChange(previousProps.result, nextProps.result)
})

function RelativeTimeText({ timestampMs }: { timestampMs: number | null }) {
  const [nowMs, setNowMs] = useState(() => Date.now())

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      setNowMs(Date.now())
    }, 1000)

    return () => {
      window.clearInterval(intervalId)
    }
  }, [])

  const label = useMemo(() => {
    if (timestampMs === null) {
      return '--'
    }

    return formatRelativeUpdateTime(new Date(timestampMs).toISOString())
  }, [nowMs, timestampMs])

  return <>{label}</>
}

function HeartbeatStateText({ status, timestampMs }: { status: TaskStatus | null; timestampMs: number | null }) {
  const [nowMs, setNowMs] = useState(() => Date.now())

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      setNowMs(Date.now())
    }, 1000)

    return () => {
      window.clearInterval(intervalId)
    }
  }, [])

  const label = useMemo(() => formatHeartbeatState(status, timestampMs, nowMs), [nowMs, status, timestampMs])
  return <>{label}</>
}

function FieldLabel({ label, hint }: { label: string; hint: string }) {
  return (
    <span className="field-label with-help">
      <span>{label}</span>
      <span className="help-chip" title={hint} aria-label={hint}>
        ?
      </span>
    </span>
  )
}

function SegmentedToggleGroup<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T
  options: ReadonlyArray<{ value: T; label: string; title: string }>
  onChange: (value: T) => void
}) {
  return (
    <div className="segmented-toggle-pair" role="radiogroup" aria-label="二选一配置">
      {options.map((option) => {
        const selected = option.value === value
        return (
          <button
            key={option.value}
            type="button"
            className={`segmented-option${selected ? ' selected' : ''}`}
            aria-pressed={selected}
            title={option.title}
            onClick={() => onChange(option.value)}
          >
            <span className="segmented-option-label">{option.label}</span>
            <span className="segmented-option-state">{selected ? '当前' : '切换'}</span>
          </button>
        )
      })}
    </div>
  )
}

function ToggleStateButton({
  label,
  enabled,
  title,
  onToggle,
  disabled = false,
  disabledLabel,
}: {
  label: string
  enabled: boolean
  title: string
  onToggle: () => void
  disabled?: boolean
  disabledLabel?: string
}) {
  const stateLabel = disabled ? (disabledLabel ?? '不可用') : enabled ? '已开启' : '已关闭'
  return (
    <button
      type="button"
      className={`state-toggle-button${enabled ? ' active' : ''}${disabled ? ' disabled' : ''}`}
      aria-pressed={enabled}
      title={title}
      disabled={disabled}
      onClick={onToggle}
    >
      <span className="state-toggle-label">{label}</span>
      <span className="state-toggle-state">{stateLabel}</span>
    </button>
  )
}

function GroupHeader({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div className="group-header field-span-2">
      <strong>{title}</strong>
      <span>{subtitle}</span>
    </div>
  )
}

function hasTaskSnapshotMeaningfulChange(current: TaskSnapshot | null, next: TaskSnapshot): boolean {
  if (current === null) {
    return true
  }

  if (current.task_id !== next.task_id || current.status !== next.status) {
    return true
  }

  if (
    current.progress.stage !== next.progress.stage ||
    current.progress.percent !== next.progress.percent ||
    current.progress.processed_units !== next.progress.processed_units ||
    current.progress.total_units !== next.progress.total_units ||
    current.progress.message !== next.progress.message ||
    current.progress.eta_seconds !== next.progress.eta_seconds ||
    current.progress.last_heartbeat_at !== next.progress.last_heartbeat_at ||
    current.progress.last_heartbeat_message !== next.progress.last_heartbeat_message
  ) {
    return true
  }

  if (current.error !== next.error) {
    return true
  }

  if (current.recent_logs.length !== next.recent_logs.length) {
    return true
  }

  for (let index = 0; index < current.recent_logs.length; index += 1) {
    if (current.recent_logs[index] !== next.recent_logs[index]) {
      return true
    }
  }

  if (hasViewerResultMeaningfulChange(current.result, next.result)) {
    return true
  }

  return false
}

export function TaskWorkbenchZh() {
  const [leftRailCollapsed, setLeftRailCollapsed] = useState(false)
  const [rightRailCollapsed, setRightRailCollapsed] = useState(false)
  const [task, setTask] = useState<TaskSnapshot | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploadedFile, setUploadedFile] = useState<UploadedFileInfo | null>(null)
  const [selectedMaskFile, setSelectedMaskFile] = useState<File | null>(null)
  const [uploadedMaskFile, setUploadedMaskFile] = useState<UploadedFileInfo | null>(null)
  const [historyFiles, setHistoryFiles] = useState<UploadedFileInfo[]>([])
  const [selectedHistoryInputPath, setSelectedHistoryInputPath] = useState<string>('')
  const [selectedHistoryMaskPath, setSelectedHistoryMaskPath] = useState<string>('')
  const [outputFileName, setOutputFileName] = useState('example-channel-result.png')
  const [pipelineConfig, setPipelineConfig] = useState<CreateTaskRequest['config']>(DEFAULT_PIPELINE_CONFIG)
  const [isUploading, setIsUploading] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [localPreviewUrl, setLocalPreviewUrl] = useState<string | null>(null)
  const [lastBackendSyncAt, setLastBackendSyncAt] = useState<number | null>(null)
  const timerRef = useRef<number | null>(null)
  const pollInFlightRef = useRef(false)

  function updatePreprocessConfig(updater: (current: PipelineConfig['preprocess']) => PipelineConfig['preprocess']) {
    setPipelineConfig((current) => ({ ...current, preprocess: updater(current.preprocess) }))
  }

  function updateFlowAccumulationConfig(
    updater: (current: PipelineConfig['flow_accumulation']) => PipelineConfig['flow_accumulation'],
  ) {
    setPipelineConfig((current) => ({ ...current, flow_accumulation: updater(current.flow_accumulation) }))
  }

  function updateFlowDirectionConfig(
    updater: (current: PipelineConfig['flow_direction']) => PipelineConfig['flow_direction'],
  ) {
    setPipelineConfig((current) => ({ ...current, flow_direction: updater(current.flow_direction) }))
  }

  function updateChannelExtractConfig(
    updater: (current: PipelineConfig['channel_extract']) => PipelineConfig['channel_extract'],
  ) {
    setPipelineConfig((current) => ({ ...current, channel_extract: updater(current.channel_extract) }))
  }

  function updateRuntimeConfig(updater: (current: PipelineConfig) => PipelineConfig) {
    setPipelineConfig((current) => updater(current))
  }

  useEffect(() => {
    void (async () => {
      try {
        const files = await listUploadedFiles()
        setHistoryFiles(files)
      } catch (error) {
        setErrorMessage((error as Error).message)
      }
    })()
  }, [])

  useEffect(() => {
    return () => {
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current)
      }
    }
  }, [])

  useEffect(() => {
    if (selectedFile === null) {
      setLocalPreviewUrl((current) => {
        if (current !== null) {
          URL.revokeObjectURL(current)
        }
        return null
      })
      return
    }

    const objectUrl = URL.createObjectURL(selectedFile)
    setLocalPreviewUrl((current) => {
      if (current !== null) {
        URL.revokeObjectURL(current)
      }
      return objectUrl
    })

    return () => {
      URL.revokeObjectURL(objectUrl)
    }
  }, [selectedFile])

  useEffect(() => {
    if (task === null || !['queued', 'running'].includes(task.status)) {
      return
    }

    timerRef.current = window.setTimeout(async () => {
      if (pollInFlightRef.current) {
        return
      }

      pollInFlightRef.current = true
      try {
        const snapshot = await getTask(task.task_id)
        setLastBackendSyncAt(Date.now())
        setTask((current) => (hasTaskSnapshotMeaningfulChange(current, snapshot) ? snapshot : current))
      } catch (error) {
        setErrorMessage((error as Error).message)
      } finally {
        pollInFlightRef.current = false
      }
    }, TASK_POLL_INTERVAL_MS)

    return () => {
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current)
      }
    }
  }, [task])

  const outputPath = useMemo(() => buildOutputPath(outputFileName), [outputFileName])
  const thresholdValue = pipelineConfig.channel_extract.accumulation_threshold
  const progressPercent = task?.progress.percent ?? 0
  const heartbeatTimestampMs = task ? parseTimestampMs(task.progress.last_heartbeat_at) : null
  const backendUpdatedAtMs = task ? parseTimestampMs(task.updated_at) : null
  const lastHeartbeatMessage = task?.progress.last_heartbeat_message || task?.progress.message || '等待第一条心跳。'
  const recentStageEvents = useMemo(
    () => [...(task?.recent_logs ?? [])].slice(-3).reverse(),
    [task?.recent_logs],
  )

  const historyInputFile = useMemo(
    () => historyFiles.find((item) => item.stored_path === selectedHistoryInputPath) ?? null,
    [historyFiles, selectedHistoryInputPath],
  )
  const historyMaskFile = useMemo(
    () => historyFiles.find((item) => item.stored_path === selectedHistoryMaskPath) ?? null,
    [historyFiles, selectedHistoryMaskPath],
  )
  const activeInputFile = uploadedFile ?? historyInputFile
  const activeMaskFile = uploadedMaskFile ?? historyMaskFile
  const inputSourceSummary = selectedFile?.name ?? activeInputFile?.filename ?? '待选择'
  const maskSourceSummary = pipelineConfig.preprocess.use_mask
    ? selectedMaskFile?.name ?? activeMaskFile?.filename ?? '待选择'
    : '未启用'

  const viewerResult = useMemo<ArtifactViewerResult | null>(() => {
    if (task?.result !== null && task?.result !== undefined) {
      return task.result
    }

    if (selectedFile !== null && localPreviewUrl !== null) {
      return {
        task_directory: '本地预览（尚未启动任务）',
        metadata_path: null,
        input_preview: localPreviewUrl,
        auto_mask: null,
        terrain_preprocessed: null,
        flow_direction: null,
        flow_accumulation: null,
        channel_mask: null,
        artifacts: {
          input_preview: {
            key: 'input_preview',
            label: '输入图像',
            stage: 'io',
            status: 'ready',
            path: localPreviewUrl,
            preview_path: localPreviewUrl,
            previewable: true,
            width: null,
            height: null,
          },
          auto_mask: { key: 'auto_mask', label: '自动遮罩', stage: 'preprocess', status: 'pending', path: null, preview_path: null, previewable: true, width: null, height: null },
          terrain_preprocessed: { key: 'terrain_preprocessed', label: '预处理', stage: 'preprocess', status: 'pending', path: null, preview_path: null, previewable: true, width: null, height: null },
          flow_direction: { key: 'flow_direction', label: '流向', stage: 'flow_direction', status: 'pending', path: null, preview_path: null, previewable: true, width: null, height: null },
          flow_accumulation: { key: 'flow_accumulation', label: '汇流累积', stage: 'flow_accumulation', status: 'pending', path: null, preview_path: null, previewable: true, width: null, height: null },
          channel_mask: { key: 'channel_mask', label: '河道结果', stage: 'channel_extract', status: 'pending', path: null, preview_path: null, previewable: true, width: null, height: null },
        },
      }
    }

    if (activeInputFile === null) {
      return null
    }

    return {
      task_directory: '历史输入预览（尚未启动任务）',
      metadata_path: null,
      input_preview: activeInputFile.stored_path,
      auto_mask: null,
      terrain_preprocessed: null,
      flow_direction: null,
      flow_accumulation: null,
      channel_mask: null,
      artifacts: {
        input_preview: {
          key: 'input_preview',
          label: '输入图像',
          stage: 'io',
          status: 'ready',
          path: activeInputFile.stored_path,
          preview_path: activeInputFile.stored_path,
          previewable: true,
          width: null,
          height: null,
        },
        auto_mask: { key: 'auto_mask', label: '自动遮罩', stage: 'preprocess', status: 'pending', path: null, preview_path: null, previewable: true, width: null, height: null },
        terrain_preprocessed: { key: 'terrain_preprocessed', label: '预处理', stage: 'preprocess', status: 'pending', path: null, preview_path: null, previewable: true, width: null, height: null },
        flow_direction: { key: 'flow_direction', label: '流向', stage: 'flow_direction', status: 'pending', path: null, preview_path: null, previewable: true, width: null, height: null },
        flow_accumulation: { key: 'flow_accumulation', label: '汇流累积', stage: 'flow_accumulation', status: 'pending', path: null, preview_path: null, previewable: true, width: null, height: null },
        channel_mask: { key: 'channel_mask', label: '河道结果', stage: 'channel_extract', status: 'pending', path: null, preview_path: null, previewable: true, width: null, height: null },
      },
    }
  }, [activeInputFile, localPreviewUrl, selectedFile, task])

  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null
    setSelectedFile(file)
    setUploadedFile(null)
    setSelectedHistoryInputPath('')
    setErrorMessage(null)
  }

  function handleMaskFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null
    setSelectedMaskFile(file)
    setUploadedMaskFile(null)
    setSelectedHistoryMaskPath('')
    setErrorMessage(null)
    updatePreprocessConfig((current) => ({ ...current, use_mask: file !== null }))
  }

  function handleHistoryInputChange(event: React.ChangeEvent<HTMLSelectElement>) {
    const nextPath = event.target.value
    setSelectedHistoryInputPath(nextPath)
    if (nextPath !== '') {
      setSelectedFile(null)
      setUploadedFile(null)
    }
    setErrorMessage(null)
  }

  function handleHistoryMaskChange(event: React.ChangeEvent<HTMLSelectElement>) {
    const nextPath = event.target.value
    setSelectedHistoryMaskPath(nextPath)
    if (nextPath !== '') {
      setSelectedMaskFile(null)
      setUploadedMaskFile(null)
      updatePreprocessConfig((current) => ({ ...current, use_mask: true }))
    }
    setErrorMessage(null)
  }

  async function handleUploadFile() {
    if (selectedFile === null) {
      setErrorMessage('请先选择一个输入文件。')
      return
    }

    setIsUploading(true)
    setErrorMessage(null)
    try {
      const result = await uploadInputFile(selectedFile)
      setUploadedFile(result)
      setHistoryFiles((current) => [result, ...current.filter((item) => item.stored_path !== result.stored_path)])
    } catch (error) {
      setErrorMessage((error as Error).message)
    } finally {
      setIsUploading(false)
    }
  }

  async function handleUploadMaskFile() {
    if (selectedMaskFile === null) {
      setErrorMessage('请先选择一个用户遮罩文件。')
      return
    }

    setIsUploading(true)
    setErrorMessage(null)
    try {
      const result = await uploadInputFile(selectedMaskFile)
      setUploadedMaskFile(result)
      setHistoryFiles((current) => [result, ...current.filter((item) => item.stored_path !== result.stored_path)])
      updatePreprocessConfig((current) => ({ ...current, use_mask: true }))
    } catch (error) {
      setErrorMessage((error as Error).message)
    } finally {
      setIsUploading(false)
    }
  }

  async function handleCreateTask() {
    if (activeInputFile === null) {
      setErrorMessage('请先上传主图，或从历史素材中选择一个输入文件。')
      return
    }

    if (pipelineConfig.preprocess.use_mask && activeMaskFile === null) {
      setErrorMessage('当前已开启用户遮罩，但还没有可用的用户遮罩文件。')
      return
    }

    setErrorMessage(null)
    setIsSubmitting(true)
    const payload: CreateTaskRequest = {
      input_path: activeInputFile.stored_path,
      mask_path: pipelineConfig.preprocess.use_mask && activeMaskFile !== null ? activeMaskFile.stored_path : null,
      output_path: outputPath,
      config: pipelineConfig,
    }

    try {
      const taskId = await createTask(payload)
      const snapshot = await getTask(taskId)
      setLastBackendSyncAt(Date.now())
      setTask(snapshot)
    } catch (error) {
      setErrorMessage((error as Error).message)
    } finally {
      setIsSubmitting(false)
    }
  }

  async function handleCancelTask() {
    if (task === null) {
      return
    }

    setErrorMessage(null)
    try {
      const snapshot = await cancelTask(task.task_id)
      setLastBackendSyncAt(Date.now())
      setTask(snapshot)
    } catch (error) {
      setErrorMessage((error as Error).message)
    }
  }

  return (
    <main className="app-shell cockpit-shell">
      <section className="top-status-bar">
        <div className="top-status-title">
          <p className="eyebrow">河道提取 MVP</p>
          <h1>大尺寸栅格河道工作台</h1>
        </div>
        <div className="top-status-chips">
          <span>界面 2026-04-14</span>
          <span>三栏折叠版</span>
          <span>状态 {task?.status ?? '空闲'}</span>
          <span>阶段 {task?.progress.stage ?? '等待中'}</span>
          <span>进度 {progressPercent.toFixed(1)}%</span>
          <span>心跳 <RelativeTimeText timestampMs={heartbeatTimestampMs} /></span>
          <span>阈值 {thresholdValue}</span>
          <span>自动遮罩 {pipelineConfig.preprocess.use_auto_mask ? '开' : '关'}</span>
          <span>用户遮罩 {pipelineConfig.preprocess.use_mask ? '开' : '关'}</span>
        </div>
      </section>

      <section className="workspace-shell">
        <aside className={`side-rail left-rail${leftRailCollapsed ? ' collapsed' : ''}`}>
          <div className="rail-header">
            <div>
              <h2>控制</h2>
              {!leftRailCollapsed ? <p className="micro-note">输入、参数与任务启动。</p> : null}
            </div>
            <button
              type="button"
              className="secondary-button compact-button"
              onClick={() => setLeftRailCollapsed((current) => !current)}
              title={leftRailCollapsed ? '展开左侧控制栏' : '折叠左侧控制栏'}
            >
              {leftRailCollapsed ? '展开' : '折叠'}
            </button>
          </div>

          {!leftRailCollapsed ? (
            <article className="panel control-panel rail-panel">
              <div className="panel-title-row">
                <h2>任务控制</h2>
                <div className="button-row inline-actions control-toolbar">
                  <button className="secondary-button compact-button" title={HELP_TEXT.uploadMain} onClick={handleUploadFile} disabled={isUploading}>
                    {isUploading ? '上传中...' : '上传主图'}
                  </button>
                  <button
                    className="secondary-button compact-button"
                    title={HELP_TEXT.uploadMask}
                    onClick={handleUploadMaskFile}
                    disabled={isUploading || selectedMaskFile === null}
                  >
                    {isUploading && selectedMaskFile !== null ? '上传中...' : '上传遮罩'}
                  </button>
                  <button className="primary-button compact-button" title={HELP_TEXT.startTask} onClick={handleCreateTask} disabled={isSubmitting}>
                    {isSubmitting ? '启动中...' : '启动任务'}
                  </button>
                  <button
                    className="secondary-button compact-button"
                    title={HELP_TEXT.cancelTask}
                    onClick={handleCancelTask}
                    disabled={task === null || !['queued', 'running'].includes(task.status)}
                  >
                    取消
                  </button>
                </div>
              </div>

              <div className="dense-form-grid compact-grid-v2">
            <GroupHeader title="输入输出" subtitle="主图、遮罩、历史素材与输出文件。" />
            <div className="field compact-field">
              <FieldLabel label="输入主图" hint={HELP_TEXT.input} />
              <input title={HELP_TEXT.input} type="file" accept=".pgm,.png,.jpg,.jpeg,.bmp,.tif,.tiff" onChange={handleFileChange} />
              <p className="micro-note">
                {selectedFile ? `已选 ${selectedFile.name} · ${formatFileSize(selectedFile.size)}` : '尚未选择输入文件'}
              </p>
            </div>

            <div className="field compact-field">
              <FieldLabel label="历史主图素材" hint={HELP_TEXT.historyInput} />
              <select title={HELP_TEXT.historyInput} value={selectedHistoryInputPath} onChange={handleHistoryInputChange}>
                <option value="">不使用历史素材</option>
                {historyFiles.map((item) => (
                  <option key={item.stored_path} value={item.stored_path}>
                    {item.filename} · {formatFileSize(item.size_bytes)}
                  </option>
                ))}
              </select>
              <p className="micro-note">直接复用已上传素材。</p>
            </div>

            <div className="field compact-field">
              <FieldLabel label="可选用户遮罩图层" hint={HELP_TEXT.mask} />
              <input title={HELP_TEXT.mask} type="file" accept=".pgm,.png,.jpg,.jpeg,.bmp,.tif,.tiff" onChange={handleMaskFileChange} />
              <p className="micro-note">
                {selectedMaskFile
                  ? `已选 ${selectedMaskFile.name} · ${formatFileSize(selectedMaskFile.size)}`
                  : '未提供用户遮罩图层，默认不会额外限制计算区域。'}
              </p>
            </div>

            <div className="field compact-field">
              <FieldLabel label="历史用户遮罩素材" hint={HELP_TEXT.historyMask} />
              <select title={HELP_TEXT.historyMask} value={selectedHistoryMaskPath} onChange={handleHistoryMaskChange}>
                <option value="">不使用历史遮罩</option>
                {historyFiles.map((item) => (
                  <option key={`mask-${item.stored_path}`} value={item.stored_path}>
                    {item.filename} · {formatFileSize(item.size_bytes)}
                  </option>
                ))}
              </select>
              <p className="micro-note">选中后自动启用“用户遮罩”。</p>
            </div>

            <div className="field compact-field">
              <FieldLabel label="当前主图来源" hint={HELP_TEXT.input} />
              <p className="mono-copy micro-note">{inputSourceSummary}</p>
            </div>

            <div className="field compact-field">
              <FieldLabel label="当前用户遮罩来源" hint={HELP_TEXT.mask} />
              <p className="mono-copy micro-note">{maskSourceSummary}</p>
            </div>

            <div className="field compact-field">
              <FieldLabel label="输出文件" hint={HELP_TEXT.output} />
              <input
                title={HELP_TEXT.output}
                type="text"
                value={outputFileName}
                onChange={(event) => setOutputFileName(event.target.value)}
                placeholder="输出文件名"
              />
            </div>

            <div className="field compact-field">
              <FieldLabel label="输出路径" hint={HELP_TEXT.output} />
              <p className="mono-copy micro-note">{outputPath}</p>
            </div>

            <GroupHeader title="预处理" subtitle="高程解释、平滑、无出口平坡修复与数据有效区。" />

            <div className="field compact-field field-span-2">
              <FieldLabel label="高低映射与开关" hint={HELP_TEXT.mapping} />
              <div className="mapping-toggle-layout">
                <SegmentedToggleGroup
                  value={pipelineConfig.preprocess.height_mapping}
                  options={[
                    { value: 'bright_is_high', label: '亮高', title: HELP_TEXT.mapping },
                    { value: 'dark_is_high', label: '暗高', title: HELP_TEXT.mapping },
                  ]}
                  onChange={(nextValue) =>
                    updatePreprocessConfig((current) => ({ ...current, height_mapping: nextValue }))
                  }
                />
                <div className="state-toggle-grid">
                  <ToggleStateButton
                    label="平滑"
                    enabled={pipelineConfig.preprocess.smooth}
                    title={HELP_TEXT.smooth}
                    onToggle={() =>
                      updatePreprocessConfig((current) => ({
                        ...current,
                        smooth: !current.smooth,
                      }))
                    }
                  />
                  <ToggleStateButton
                    label="NoData"
                    enabled={pipelineConfig.preprocess.preserve_nodata}
                    title={HELP_TEXT.nodata}
                    onToggle={() =>
                      updatePreprocessConfig((current) => ({
                        ...current,
                        preserve_nodata: !current.preserve_nodata,
                      }))
                    }
                  />
                  <ToggleStateButton
                    label="用户遮罩"
                    enabled={pipelineConfig.preprocess.use_mask}
                    title={HELP_TEXT.maskToggle}
                    disabled={selectedMaskFile === null && historyMaskFile === null && uploadedMaskFile === null}
                    disabledLabel="需先选遮罩"
                    onToggle={() =>
                      updatePreprocessConfig((current) => ({
                        ...current,
                        use_mask: !current.use_mask,
                      }))
                    }
                  />
                  <ToggleStateButton
                    label="自动遮罩"
                    enabled={pipelineConfig.preprocess.use_auto_mask}
                    title={HELP_TEXT.autoMask}
                    onToggle={() =>
                      updatePreprocessConfig((current) => ({
                        ...current,
                        use_auto_mask: !current.use_auto_mask,
                      }))
                    }
                  />
                  <ToggleStateButton
                    label="Rust D8"
                    enabled={pipelineConfig.flow_direction.use_rust_kernel}
                    title={HELP_TEXT.rustKernel}
                    onToggle={() =>
                      updateFlowDirectionConfig((current) => ({
                        ...current,
                        use_rust_kernel: !current.use_rust_kernel,
                      }))
                    }
                  />
                </div>
                <p className="micro-note">
                  自动遮罩会独立参与后续计算；用户遮罩只有在开启“用户遮罩”且已提供图层时才会额外参与。
                </p>
              </div>
            </div>

            <div className="field compact-field">
              <FieldLabel label="边界敏感度" hint={HELP_TEXT.autoMaskBorder} />
              <div className="range-with-input compact-tight">
                <input
                  title={HELP_TEXT.autoMaskBorder}
                  type="range"
                  min={0.1}
                  max={5}
                  step={0.1}
                  value={pipelineConfig.preprocess.auto_mask_border_sensitivity}
                  onChange={(event) =>
                    updatePreprocessConfig((current) => ({
                      ...current,
                      auto_mask_border_sensitivity: Number(event.target.value),
                    }))
                  }
                />
                <input
                  title={HELP_TEXT.autoMaskBorder}
                  type="number"
                  min={0.1}
                  max={5}
                  step={0.1}
                  value={pipelineConfig.preprocess.auto_mask_border_sensitivity}
                  onChange={(event) =>
                    updatePreprocessConfig((current) => ({
                      ...current,
                      auto_mask_border_sensitivity: Math.max(0.1, Number(event.target.value) || 0.1),
                    }))
                  }
                />
              </div>
            </div>

            <div className="field compact-field">
              <FieldLabel label="纹理敏感度" hint={HELP_TEXT.autoMaskTexture} />
              <div className="range-with-input compact-tight">
                <input
                  title={HELP_TEXT.autoMaskTexture}
                  type="range"
                  min={0.1}
                  max={5}
                  step={0.1}
                  value={pipelineConfig.preprocess.auto_mask_texture_sensitivity}
                  onChange={(event) =>
                    updatePreprocessConfig((current) => ({
                      ...current,
                      auto_mask_texture_sensitivity: Number(event.target.value),
                    }))
                  }
                />
                <input
                  title={HELP_TEXT.autoMaskTexture}
                  type="number"
                  min={0.1}
                  max={5}
                  step={0.1}
                  value={pipelineConfig.preprocess.auto_mask_texture_sensitivity}
                  onChange={(event) =>
                    updatePreprocessConfig((current) => ({
                      ...current,
                      auto_mask_texture_sensitivity: Math.max(0.1, Number(event.target.value) || 0.1),
                    }))
                  }
                />
              </div>
            </div>

            <div className="field compact-field">
              <FieldLabel label="最小区域面积" hint={HELP_TEXT.autoMaskRegion} />
              <input
                title={HELP_TEXT.autoMaskRegion}
                type="number"
                min={1}
                step={128}
                value={pipelineConfig.preprocess.auto_mask_min_region_size}
                onChange={(event) =>
                  updatePreprocessConfig((current) => ({
                    ...current,
                    auto_mask_min_region_size: Math.max(1, Number(event.target.value) || 1),
                  }))
                }
              />
            </div>

            <div className="field compact-field">
              <FieldLabel label="手动编辑遮罩" hint={HELP_TEXT.manualMaskEdit} />
              <button type="button" className="secondary-button compact-button" disabled title={HELP_TEXT.manualMaskEdit}>
                预留
              </button>
              <p className="micro-note">后续接入前端涂抹修正。</p>
            </div>

            <div className="field compact-field">
              <FieldLabel label="无出口平坡/填洼修复" hint={HELP_TEXT.fillSinks} />
              <ToggleStateButton
                label="填洼修复"
                enabled={pipelineConfig.preprocess.fill_sinks}
                title={HELP_TEXT.fillSinks}
                onToggle={() =>
                  updatePreprocessConfig((current) => ({
                    ...current,
                    fill_sinks: !current.fill_sinks,
                  }))
                }
              />
              <p className="micro-note">关闭后更容易断流。</p>
            </div>

            <div className="field compact-field">
              <FieldLabel label="平滑核" hint={HELP_TEXT.smoothKernel} />
              <div className="range-with-input compact-tight">
                <input
                  title={HELP_TEXT.smoothKernel}
                  type="range"
                  min={1}
                  max={15}
                  step={2}
                  value={pipelineConfig.preprocess.smooth_kernel_size}
                  onChange={(event) =>
                    updatePreprocessConfig((current) => ({ ...current, smooth_kernel_size: Number(event.target.value) }))
                  }
                />
                <input
                  title={HELP_TEXT.smoothKernel}
                  type="number"
                  min={1}
                  max={15}
                  step={2}
                  value={pipelineConfig.preprocess.smooth_kernel_size}
                  onChange={(event) =>
                    updatePreprocessConfig((current) => ({
                      ...current,
                      smooth_kernel_size: Math.max(1, Number(event.target.value) || 1),
                    }))
                  }
                />
              </div>
            </div>

            <div className="field compact-field">
              <FieldLabel label="NoData" hint={HELP_TEXT.nodata} />
              <input
                title={HELP_TEXT.nodata}
                type="number"
                min={0}
                max={255}
                value={pipelineConfig.preprocess.nodata_value ?? ''}
                onChange={(event) =>
                  updatePreprocessConfig((current) => ({
                    ...current,
                    nodata_value: event.target.value.trim() === '' ? null : Number(event.target.value),
                  }))
                }
                placeholder="留空"
              />
            </div>

            <GroupHeader title="流向" subtitle="平坡出口与路径形态的加权策略。" />

            <div className="field compact-field">
              <FieldLabel label="坡降权重" hint={HELP_TEXT.slopeWeight} />
              <div className="range-with-input compact-tight">
                <input
                  title={HELP_TEXT.slopeWeight}
                  type="range"
                  min={0}
                  max={3}
                  step={0.05}
                  value={pipelineConfig.flow_direction.slope_weight}
                  onChange={(event) =>
                    updateFlowDirectionConfig((current) => ({
                      ...current,
                      slope_weight: Number(event.target.value),
                    }))
                  }
                />
                <input
                  title={HELP_TEXT.slopeWeight}
                  type="number"
                  min={0}
                  max={10}
                  step={0.05}
                  value={pipelineConfig.flow_direction.slope_weight}
                  onChange={(event) =>
                    updateFlowDirectionConfig((current) => ({
                      ...current,
                      slope_weight: Math.max(0, Number(event.target.value) || 0),
                    }))
                  }
                />
              </div>
            </div>

            <div className="field compact-field">
              <FieldLabel label="平坡逃逸权重" hint={HELP_TEXT.flatEscapeWeight} />
              <div className="range-with-input compact-tight">
                <input
                  title={HELP_TEXT.flatEscapeWeight}
                  type="range"
                  min={0}
                  max={3}
                  step={0.05}
                  value={pipelineConfig.flow_direction.flat_escape_weight}
                  onChange={(event) =>
                    updateFlowDirectionConfig((current) => ({
                      ...current,
                      flat_escape_weight: Number(event.target.value),
                    }))
                  }
                />
                <input
                  title={HELP_TEXT.flatEscapeWeight}
                  type="number"
                  min={0}
                  max={10}
                  step={0.05}
                  value={pipelineConfig.flow_direction.flat_escape_weight}
                  onChange={(event) =>
                    updateFlowDirectionConfig((current) => ({
                      ...current,
                      flat_escape_weight: Math.max(0, Number(event.target.value) || 0),
                    }))
                  }
                />
              </div>
            </div>

            <div className="field compact-field">
              <FieldLabel label="出口接近权重" hint={HELP_TEXT.outletProximityWeight} />
              <div className="range-with-input compact-tight">
                <input
                  title={HELP_TEXT.outletProximityWeight}
                  type="range"
                  min={0}
                  max={3}
                  step={0.05}
                  value={pipelineConfig.flow_direction.outlet_proximity_weight}
                  onChange={(event) =>
                    updateFlowDirectionConfig((current) => ({
                      ...current,
                      outlet_proximity_weight: Number(event.target.value),
                    }))
                  }
                />
                <input
                  title={HELP_TEXT.outletProximityWeight}
                  type="number"
                  min={0}
                  max={10}
                  step={0.05}
                  value={pipelineConfig.flow_direction.outlet_proximity_weight}
                  onChange={(event) =>
                    updateFlowDirectionConfig((current) => ({
                      ...current,
                      outlet_proximity_weight: Math.max(0, Number(event.target.value) || 0),
                    }))
                  }
                />
              </div>
            </div>

            <div className="field compact-field">
              <FieldLabel label="连续性权重" hint={HELP_TEXT.continuityWeight} />
              <div className="range-with-input compact-tight">
                <input
                  title={HELP_TEXT.continuityWeight}
                  type="range"
                  min={0}
                  max={3}
                  step={0.05}
                  value={pipelineConfig.flow_direction.continuity_weight}
                  onChange={(event) =>
                    updateFlowDirectionConfig((current) => ({
                      ...current,
                      continuity_weight: Number(event.target.value),
                    }))
                  }
                />
                <input
                  title={HELP_TEXT.continuityWeight}
                  type="number"
                  min={0}
                  max={10}
                  step={0.05}
                  value={pipelineConfig.flow_direction.continuity_weight}
                  onChange={(event) =>
                    updateFlowDirectionConfig((current) => ({
                      ...current,
                      continuity_weight: Math.max(0, Number(event.target.value) || 0),
                    }))
                  }
                />
              </div>
            </div>

            <div className="field compact-field">
              <FieldLabel label="出口长度权重" hint={HELP_TEXT.outletLengthWeight} />
              <div className="range-with-input compact-tight">
                <input
                  title={HELP_TEXT.outletLengthWeight}
                  type="range"
                  min={0}
                  max={3}
                  step={0.05}
                  value={pipelineConfig.flow_direction.flat_outlet_length_weight}
                  onChange={(event) =>
                    updateFlowDirectionConfig((current) => ({
                      ...current,
                      flat_outlet_length_weight: Number(event.target.value),
                    }))
                  }
                />
                <input
                  title={HELP_TEXT.outletLengthWeight}
                  type="number"
                  min={0}
                  max={10}
                  step={0.05}
                  value={pipelineConfig.flow_direction.flat_outlet_length_weight}
                  onChange={(event) =>
                    updateFlowDirectionConfig((current) => ({
                      ...current,
                      flat_outlet_length_weight: Math.max(0, Number(event.target.value) || 0),
                    }))
                  }
                />
              </div>
            </div>

            <div className="field compact-field">
              <FieldLabel label="出口距离权重" hint={HELP_TEXT.outletDistanceWeight} />
              <div className="range-with-input compact-tight">
                <input
                  title={HELP_TEXT.outletDistanceWeight}
                  type="range"
                  min={0}
                  max={5}
                  step={0.1}
                  value={pipelineConfig.flow_direction.flat_outlet_distance_weight}
                  onChange={(event) =>
                    updateFlowDirectionConfig((current) => ({
                      ...current,
                      flat_outlet_distance_weight: Number(event.target.value),
                    }))
                  }
                />
                <input
                  title={HELP_TEXT.outletDistanceWeight}
                  type="number"
                  min={0}
                  max={10}
                  step={0.1}
                  value={pipelineConfig.flow_direction.flat_outlet_distance_weight}
                  onChange={(event) =>
                    updateFlowDirectionConfig((current) => ({
                      ...current,
                      flat_outlet_distance_weight: Math.max(0, Number(event.target.value) || 0),
                    }))
                  }
                />
              </div>
            </div>

            <div className="field compact-field">
              <FieldLabel label="汇流预览" hint={HELP_TEXT.accumulationPreview} />
              <ToggleStateButton
                label="对数增强"
                enabled={pipelineConfig.flow_accumulation.normalize}
                title={HELP_TEXT.accumulationPreview}
                onToggle={() =>
                  updateFlowAccumulationConfig((current) => ({ ...current, normalize: !current.normalize }))
                }
              />
            </div>

            <div className="field compact-field">
              <FieldLabel label="保留中间产物" hint={HELP_TEXT.keepArtifacts} />
              <ToggleStateButton
                label="中间产物"
                enabled={pipelineConfig.save_intermediates}
                title={HELP_TEXT.keepArtifacts}
                onToggle={() =>
                  updateRuntimeConfig((current) => ({ ...current, save_intermediates: !current.save_intermediates }))
                }
              />
            </div>

            <div className="field compact-field">
              <GroupHeader title="汇流与输出" subtitle="预览增强、阈值提取与中间产物保留。" />
              <FieldLabel label="河道阈值" hint={HELP_TEXT.threshold} />
              <div className="range-with-input compact-tight threshold-row">
                <input
                  title={HELP_TEXT.threshold}
                  type="range"
                  min={1}
                  max={1000}
                  step={1}
                  value={thresholdValue}
                  onChange={(event) =>
                    updateChannelExtractConfig((current) => ({
                      ...current,
                      accumulation_threshold: Number(event.target.value),
                    }))
                  }
                />
                <input
                  title={HELP_TEXT.threshold}
                  type="number"
                  min={1}
                  step={1}
                  value={thresholdValue}
                  onChange={(event) =>
                    updateChannelExtractConfig((current) => ({
                      ...current,
                      accumulation_threshold: Math.max(1, Number(event.target.value) || 1),
                    }))
                  }
                />
              </div>
              <p className="micro-note">低阈值更密，高阈值更偏主干。</p>
            </div>

            <div className="field compact-field">
              <GroupHeader title="运行控制" subtitle="调节任务反馈粒度与运行开销。" />
              <FieldLabel label="进度分片" hint={HELP_TEXT.totalTiles} />
              <div className="range-with-input compact-tight threshold-row">
                <input
                  title={HELP_TEXT.totalTiles}
                  type="range"
                  min={8}
                  max={256}
                  step={8}
                  value={pipelineConfig.total_tiles}
                  onChange={(event) =>
                    updateRuntimeConfig((current) => ({ ...current, total_tiles: Number(event.target.value) }))
                  }
                />
                <input
                  title={HELP_TEXT.totalTiles}
                  type="number"
                  min={8}
                  step={8}
                  value={pipelineConfig.total_tiles}
                  onChange={(event) =>
                    updateRuntimeConfig((current) => ({
                      ...current,
                      total_tiles: Math.max(8, Number(event.target.value) || 8),
                    }))
                  }
                />
              </div>
            </div>
              </div>

              {errorMessage ? <p className="error-banner">{errorMessage}</p> : null}
            </article>
          ) : (
            <article className="panel collapsed-rail-panel">
              <button className="secondary-button compact-button rail-mini-button" type="button" onClick={handleUploadFile}>
                主图
              </button>
              <button className="secondary-button compact-button rail-mini-button" type="button" onClick={handleCreateTask}>
                启动
              </button>
            </article>
          )}
        </aside>

        <section className="center-stage">
          <MemoArtifactViewerTabs result={viewerResult} />
        </section>

        <aside className={`side-rail right-rail${rightRailCollapsed ? ' collapsed' : ''}`}>
          <div className="rail-header">
            <div>
              <h2>状态</h2>
              {!rightRailCollapsed ? <p className="micro-note">进度、步骤和结果摘要。</p> : null}
            </div>
            <button
              type="button"
              className="secondary-button compact-button"
              onClick={() => setRightRailCollapsed((current) => !current)}
              title={rightRailCollapsed ? '展开右侧信息栏' : '折叠右侧信息栏'}
            >
              {rightRailCollapsed ? '展开' : '折叠'}
            </button>
          </div>

          {!rightRailCollapsed ? (
            <div className="side-stack">
              <article className="panel compact-panel">
            <h2>任务状态</h2>
            <div className="status-strip-grid">
              <div className="metric-card compact-metric">
                <span className="field-label">状态</span>
                <strong>{task?.status ?? '空闲'}</strong>
              </div>
              <div className="metric-card compact-metric">
                <span className="field-label">阶段</span>
                <strong>{task?.progress.stage ?? '等待中'}</strong>
              </div>
              <div className="metric-card compact-metric">
                <span className="field-label">单元</span>
                <strong>{task ? `${task.progress.processed_units}/${task.progress.total_units}` : '0/0'}</strong>
              </div>
              <div className="metric-card compact-metric">
                <span className="field-label">ETA</span>
                <strong>{task?.progress.eta_seconds != null ? `${task?.progress.eta_seconds}s` : '--'}</strong>
              </div>
              <div className="metric-card compact-metric">
                <span className="field-label">活跃度</span>
                <strong><HeartbeatStateText status={task?.status ?? null} timestampMs={heartbeatTimestampMs} /></strong>
              </div>
              <div className="metric-card compact-metric">
                <span className="field-label">最近心跳</span>
                <strong><RelativeTimeText timestampMs={heartbeatTimestampMs} /></strong>
              </div>
            </div>
            <div className="task-activity-strip">
              <span className="field-label">当前子步骤</span>
              <strong>{task?.progress.message ?? '等待任务启动。'}</strong>
            </div>
            <div className="task-activity-strip">
              <span className="field-label">最近活动</span>
              <strong>
                {lastHeartbeatMessage}
                {' · '}
                <RelativeTimeText timestampMs={heartbeatTimestampMs} />
              </strong>
            </div>
            <div className="progress-block compact-progress">
              <div className="progress-meta">
                <span>阶段进度</span>
                <strong>{progressPercent.toFixed(1)}%</strong>
              </div>
              <div className="progress-track" aria-hidden="true">
                <div className="progress-fill" style={{ width: `${progressPercent}%` }} />
              </div>
              <p className="muted-copy">
                {task?.progress.processed_units ?? 0}/{task?.progress.total_units ?? 0} 单元，
                预计剩余 {task?.progress.eta_seconds != null ? `${task.progress.eta_seconds}s` : '--'}
              </p>
              <p className="muted-copy">
                后端快照 <RelativeTimeText timestampMs={backendUpdatedAtMs} />，本地同步{' '}
                <RelativeTimeText timestampMs={lastBackendSyncAt} />
              </p>
            </div>
            <div className="stage-event-list">
              <span className="field-label">最近阶段事件</span>
              {recentStageEvents.length > 0 ? (
                recentStageEvents.map((logLine) => <p key={logLine}>{logLine}</p>)
              ) : (
                <p>等待第一条阶段事件。</p>
              )}
            </div>
          </article>

          <article className="panel compact-panel">
            <h2>步骤入口</h2>
            <div className="step-entry-list dense-step-list">
              {STEP_ENTRIES.map((step) => (
                <div key={step.key} className="step-entry-card dense-step-card">
                  <div className="step-entry-copy">
                    <strong>{step.title}</strong>
                    <p className="micro-note">{step.description}</p>
                  </div>
                  <button type="button" className="secondary-button compact-button" disabled>
                    预留
                  </button>
                </div>
              ))}
            </div>
          </article>

          <article className="panel compact-panel">
            <h2>日志与产物</h2>
            <div className="compact-meta-list">
              <p><strong>任务目录</strong><span className="mono-copy">{task?.result?.task_directory ?? '待生成'}</span></p>
              <p><strong>元数据</strong><span className="mono-copy">{task?.result?.metadata_path ?? '待生成'}</span></p>
              <p><strong>最终输出</strong><span className="mono-copy">{task?.result?.channel_mask ?? outputPath}</span></p>
              <p><strong>主图来源</strong><span className="mono-copy">{activeInputFile?.stored_path ?? '未选择'}</span></p>
              <p><strong>用户遮罩</strong><span className="mono-copy">{pipelineConfig.preprocess.use_mask ? activeMaskFile?.stored_path ?? '待选择' : '未启用'}</span></p>
            </div>
            <div className="log-panel compact-log-panel">
              {(task?.recent_logs ?? ['等待第一条任务日志。']).map((logLine) => (
                <p key={logLine}>{logLine}</p>
              ))}
            </div>
          </article>
            </div>
          ) : (
            <article className="panel collapsed-rail-panel">
              <span className="rail-kpi">{task?.status ?? '空闲'}</span>
              <span className="rail-kpi">{task?.progress.stage ?? '等待中'}</span>
              <span className="rail-kpi">{progressPercent.toFixed(0)}%</span>
            </article>
          )}
        </aside>
      </section>

      <section className="bottom-info-bar">
        <div className="bottom-info-group">
          <strong>主图</strong>
          <span className="mono-copy">{activeInputFile?.stored_path ?? '未选择'}</span>
        </div>
        <div className="bottom-info-group">
          <strong>输出</strong>
          <span className="mono-copy">{task?.result?.channel_mask ?? outputPath}</span>
        </div>
        <div className="bottom-info-group">
          <strong>目录</strong>
          <span className="mono-copy">{task?.result?.task_directory ?? '待生成'}</span>
        </div>
        <div className="bottom-info-group bottom-log-group">
          <strong>最新日志</strong>
          <span>{task?.recent_logs?.[task.recent_logs.length - 1] ?? '等待第一条任务日志。'}</span>
        </div>
      </section>
    </main>
  )
}
