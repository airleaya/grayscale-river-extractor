import { memo, useEffect, useMemo, useRef, useState } from 'react'
import {
  cancelTask,
  continueTask,
  createDraftTask,
  deleteTask,
  deleteUploadedFile,
  getTask,
  listTasks,
  listUploadedFiles,
  pauseTask,
  renameTask,
  renameUploadedFile,
  rerunTask,
  resumeTask,
  startExistingTask,
  updateDraftTaskState,
  uploadInputFile,
} from './api'
import './App.css'
import './App.cockpit.css'
import { ArtifactViewerTabs } from './ArtifactViewerTabs'
import { DEFAULT_PIPELINE_CONFIG } from './constants'
import type {
  ArtifactViewerResult,
  CreateTaskRequest,
  DraftTaskState,
  PipelineConfig,
  PipelineStage,
  TaskSnapshot,
  TaskStatus,
  UploadedFileInfo,
  UploadedFileKind,
} from './types'

const TASK_POLL_INTERVAL_MS = 1600
const ACTIVE_TASK_STORAGE_KEY = 'river.activeTaskId'
const ARTIFACT_KEYS = [
  'input_preview',
  'user_mask',
  'auto_mask',
  'terrain_preprocessed',
  'flow_direction',
  'flow_accumulation',
  'channel_mask',
] as const
const STEP_ENTRIES: ReadonlyArray<{ key: PipelineStage; title: string; description: string }> = [
  { key: 'io', title: '输入检查', description: '校验输入栅格、生成输入预览，并建立任务目录。' },
  { key: 'preprocess', title: '预处理', description: '高低映射、平滑、填洼、自动遮罩与用户遮罩。' },
  { key: 'flow_direction', title: '流向', description: '计算 D8 流向并修补残余无流向像素。' },
  { key: 'flow_accumulation', title: '汇流', description: '构建汇流依赖图并传播累积结果。' },
  { key: 'channel_extract', title: '河道', description: '提取河道、边缘终止并过滤短河道。' },
]
const PIPELINE_STAGE_ORDER: PipelineStage[] = STEP_ENTRIES.map((entry) => entry.key)
type DraftSyncState = 'idle' | 'pending' | 'saving' | 'saved' | 'error'

function buildOutputPath(fileName: string): string {
  const trimmed = fileName.trim()
  const safeName = trimmed.length > 0 ? trimmed : 'river-output.png'
  return `data/output/${safeName}`
}

function extractOutputFileName(outputPath: string | null | undefined): string {
  if (!outputPath) {
    return 'example-channel-result.png'
  }

  const normalizedPath = outputPath.replace(/\\/g, '/')
  const lastSegment = normalizedPath.split('/').pop()
  return lastSegment && lastSegment.trim().length > 0 ? lastSegment : 'example-channel-result.png'
}

function buildDraftStateSignature(draftState: DraftTaskState): string {
  return JSON.stringify(draftState)
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

function parseTimestampMs(isoTimestamp: string | null): number | null {
  if (isoTimestamp === null) {
    return null
  }

  const timestampMs = Date.parse(isoTimestamp)
  return Number.isNaN(timestampMs) ? null : timestampMs
}

function formatHeartbeatState(status: TaskStatus | null, timestampMs: number | null, nowMs: number): string {
  if (status === null || !['queued', 'running', 'pausing'].includes(status)) {
    return status === 'draft' ? '草稿' : '空闲'
  }

  if (status === 'queued') {
    if (timestampMs === null) {
      return '排队中'
    }

    const deltaSeconds = Math.max(0, Math.round((nowMs - timestampMs) / 1000))
    return deltaSeconds <= 10 ? '排队中' : '排队等待'
  }

  if (status === 'pausing') {
    return '暂停中'
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

function normalizeInheritedStageSelection(
  requestedStages: PipelineStage[],
  availableStages: Set<PipelineStage>,
): PipelineStage[] {
  const requestedStageSet = new Set(requestedStages)
  const normalizedStages: PipelineStage[] = []

  for (const stage of PIPELINE_STAGE_ORDER) {
    if (!availableStages.has(stage) || !requestedStageSet.has(stage)) {
      break
    }
    normalizedStages.push(stage)
  }

  return normalizedStages
}

function buildEmptyArtifact(key: (typeof ARTIFACT_KEYS)[number], stage: PipelineStage, label: string) {
  return {
    key,
    label,
    stage,
    status: 'pending' as const,
    path: null,
    preview_path: null,
    previewable: true,
    width: null,
    height: null,
  }
}

function buildPreviewResult(
  inputPreviewPath: string | null,
  maskPreviewPath: string | null,
  taskDirectory: string,
): ArtifactViewerResult | null {
  if (inputPreviewPath === null && maskPreviewPath === null) {
    return null
  }

  return {
    task_directory: taskDirectory,
    metadata_path: null,
    input_preview: inputPreviewPath,
    auto_mask: null,
    terrain_preprocessed: null,
    flow_direction: null,
    flow_accumulation: null,
    channel_mask: null,
    artifacts: {
      input_preview:
        inputPreviewPath !== null
          ? {
              key: 'input_preview',
              label: '输入图像',
              stage: 'io',
              status: 'ready',
              path: inputPreviewPath,
              preview_path: inputPreviewPath,
              previewable: true,
              width: null,
              height: null,
            }
          : buildEmptyArtifact('input_preview', 'io', '输入图像'),
      user_mask:
        maskPreviewPath !== null
          ? {
              key: 'user_mask',
              label: '用户遮罩',
              stage: 'preprocess',
              status: 'ready',
              path: maskPreviewPath,
              preview_path: maskPreviewPath,
              previewable: true,
              width: null,
              height: null,
            }
          : buildEmptyArtifact('user_mask', 'preprocess', '用户遮罩'),
      auto_mask: buildEmptyArtifact('auto_mask', 'preprocess', '自动遮罩'),
      terrain_preprocessed: buildEmptyArtifact('terrain_preprocessed', 'preprocess', '预处理'),
      flow_direction: buildEmptyArtifact('flow_direction', 'flow_direction', '流向'),
      flow_accumulation: buildEmptyArtifact('flow_accumulation', 'flow_accumulation', '汇流累积'),
      channel_mask: buildEmptyArtifact('channel_mask', 'channel_extract', '河道结果'),
    },
  }
}

function RelativeTimeText({ timestampMs }: { timestampMs: number | null }) {
  const [nowMs, setNowMs] = useState(() => Date.now())

  useEffect(() => {
    const intervalId = window.setInterval(() => setNowMs(Date.now()), 1000)
    return () => window.clearInterval(intervalId)
  }, [])

  const label = useMemo(() => {
    if (timestampMs === null) {
      return '--'
    }

    const deltaSeconds = Math.max(0, Math.round((nowMs - timestampMs) / 1000))
    if (deltaSeconds < 60) {
      return `${deltaSeconds}s 前`
    }

    if (deltaSeconds < 3600) {
      return `${Math.floor(deltaSeconds / 60)}m 前`
    }

    return `${Math.floor(deltaSeconds / 3600)}h 前`
  }, [nowMs, timestampMs])

  return <>{label}</>
}

function HeartbeatStateText({ status, timestampMs }: { status: TaskStatus | null; timestampMs: number | null }) {
  const [nowMs, setNowMs] = useState(() => Date.now())

  useEffect(() => {
    const intervalId = window.setInterval(() => setNowMs(Date.now()), 1000)
    return () => window.clearInterval(intervalId)
  }, [])

  return <>{formatHeartbeatState(status, timestampMs, nowMs)}</>
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

function GroupHeader({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div className="group-header field-span-2">
      <strong>{title}</strong>
      <span>{subtitle}</span>
    </div>
  )
}

function SegmentedToggleGroup<T extends string>({
  value,
  options,
  onChange,
  disabled = false,
}: {
  value: T
  options: ReadonlyArray<{ value: T; label: string; title: string }>
  onChange: (value: T) => void
  disabled?: boolean
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
            disabled={disabled}
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

function ManagedFileList({
  title,
  emptyText,
  files,
  selectedPath,
  onSelect,
  onRename,
  onDelete,
  disabled,
}: {
  title: string
  emptyText: string
  files: UploadedFileInfo[]
  selectedPath: string
  onSelect: (path: string) => void
  onRename: (file: UploadedFileInfo) => void
  onDelete: (file: UploadedFileInfo) => void
  disabled: boolean
}) {
  return (
    <div className="field compact-field field-span-2">
      <FieldLabel label={title} hint={emptyText} />
      <div className="managed-library-list">
        {files.length > 0 ? (
          files.map((file) => (
            <div
              key={file.stored_path}
              className={`managed-library-item${selectedPath === file.stored_path ? ' selected' : ''}`}
            >
              <button type="button" className="managed-library-select" onClick={() => onSelect(file.stored_path)} disabled={disabled}>
                <strong>{file.filename}</strong>
                <span>{formatFileSize(file.size_bytes)}</span>
                <span className="mono-copy">{file.stored_path}</span>
              </button>
              <div className="managed-library-actions">
                <button type="button" className="secondary-button compact-button" onClick={() => onRename(file)} disabled={disabled}>
                  重命名
                </button>
                <button type="button" className="secondary-button compact-button" onClick={() => onDelete(file)} disabled={disabled}>
                  删除
                </button>
              </div>
            </div>
          ))
        ) : (
          <p className="micro-note">{emptyText}</p>
        )}
      </div>
    </div>
  )
}

const MemoArtifactViewerTabs = memo(ArtifactViewerTabs)

export function TaskWorkbenchZh() {
  const [leftRailCollapsed, setLeftRailCollapsed] = useState(false)
  const [rightRailCollapsed, setRightRailCollapsed] = useState(false)
  const [tasks, setTasks] = useState<TaskSnapshot[]>([])
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [selectedMaskFile, setSelectedMaskFile] = useState<File | null>(null)
  const [uploadedInputFile, setUploadedInputFile] = useState<UploadedFileInfo | null>(null)
  const [uploadedMaskFile, setUploadedMaskFile] = useState<UploadedFileInfo | null>(null)
  const [inputFiles, setInputFiles] = useState<UploadedFileInfo[]>([])
  const [maskFiles, setMaskFiles] = useState<UploadedFileInfo[]>([])
  const [selectedHistoryInputPath, setSelectedHistoryInputPath] = useState('')
  const [selectedHistoryMaskPath, setSelectedHistoryMaskPath] = useState('')
  const [outputFileName, setOutputFileName] = useState('example-channel-result.png')
  const [pipelineConfig, setPipelineConfig] = useState<CreateTaskRequest['config']>(DEFAULT_PIPELINE_CONFIG)
  const [inheritIntermediates, setInheritIntermediates] = useState(true)
  const [inheritStageOutputsByTaskId, setInheritStageOutputsByTaskId] = useState<Record<string, PipelineStage[]>>({})
  const [isUploadingInput, setIsUploadingInput] = useState(false)
  const [isUploadingMask, setIsUploadingMask] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isTaskActionPending, setIsTaskActionPending] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [draftSyncState, setDraftSyncState] = useState<DraftSyncState>('idle')
  const [draftSyncError, setDraftSyncError] = useState<string | null>(null)
  const [lastDraftSavedAt, setLastDraftSavedAt] = useState<number | null>(null)
  const [localPreviewUrl, setLocalPreviewUrl] = useState<string | null>(null)
  const [localMaskPreviewUrl, setLocalMaskPreviewUrl] = useState<string | null>(null)
  const [lastBackendSyncAt, setLastBackendSyncAt] = useState<number | null>(null)
  const timerRef = useRef<number | null>(null)
  const draftSaveTimerRef = useRef<number | null>(null)
  const pollInFlightRef = useRef(false)
  const inputUploadTokenRef = useRef(0)
  const maskUploadTokenRef = useRef(0)
  const isApplyingDraftStateRef = useRef(false)
  const lastDraftStateSignatureRef = useRef<string>('')

  const task = useMemo(
    () => tasks.find((item) => item.task_id === selectedTaskId) ?? null,
    [selectedTaskId, tasks],
  )
  const activeRunningTask = useMemo(
    () => tasks.find((item) => ['queued', 'running', 'pausing'].includes(item.status)) ?? null,
    [tasks],
  )
  const historyInputFile = useMemo(
    () => inputFiles.find((item) => item.stored_path === selectedHistoryInputPath) ?? null,
    [inputFiles, selectedHistoryInputPath],
  )
  const historyMaskFile = useMemo(
    () => maskFiles.find((item) => item.stored_path === selectedHistoryMaskPath) ?? null,
    [maskFiles, selectedHistoryMaskPath],
  )
  const activeInputFile = uploadedInputFile ?? historyInputFile
  const activeMaskFile = uploadedMaskFile ?? historyMaskFile
  const outputPath = useMemo(() => buildOutputPath(outputFileName), [outputFileName])
  const isTaskActive = task !== null && ['queued', 'running', 'pausing'].includes(task.status)
  const canPauseTask = task !== null && ['queued', 'running'].includes(task.status)
  const canResumeTask = task?.status === 'paused'
  const canContinueSelectedTask = task !== null && ['paused', 'completed', 'failed', 'canceled'].includes(task.status)
  const canRerunSelectedTask = task !== null && !['draft', 'queued', 'running', 'pausing'].includes(task.status)
  const heartbeatTimestampMs = task ? parseTimestampMs(task.progress.last_heartbeat_at) : null
  const backendUpdatedAtMs = task ? parseTimestampMs(task.updated_at) : null
  const completedStageSet = useMemo(() => new Set(task?.completed_stages ?? []), [task?.completed_stages])
  const inheritStageOutputs = useMemo(() => {
    if (task === null) {
      return []
    }

    const requestedStages = inheritStageOutputsByTaskId[task.task_id] ?? task.completed_stages
    return normalizeInheritedStageSelection(requestedStages, completedStageSet)
  }, [completedStageSet, inheritStageOutputsByTaskId, task])
  const inheritedStageSet = useMemo(() => new Set(inheritStageOutputs), [inheritStageOutputs])
  const inheritedStageSummary = inheritStageOutputs.length > 0
    ? inheritStageOutputs
        .map((stage) => STEP_ENTRIES.find((entry) => entry.key === stage)?.title ?? stage)
        .join(' / ')
    : '无，继续时将从头重算。'
  const leftLockReason = useMemo(() => {
    if (task === null) {
      return '请先在右侧新建并选中一个任务。'
    }

    if (task.status !== 'draft') {
      return '当前选中的是历史任务。若要修改左侧参数，请先在右侧新建一个草稿任务。'
    }

    if (activeRunningTask !== null) {
      return `任务进行中：${activeRunningTask.name}。运行结束前左侧配置暂时锁定。`
    }

    return null
  }, [activeRunningTask, task])
  const leftControlsDisabled = leftLockReason !== null || isSubmitting || isTaskActionPending || isUploadingInput || isUploadingMask
  const progressPercent = task?.progress.percent ?? 0
  const lastHeartbeatMessage = task?.progress.last_heartbeat_message || task?.progress.message || '等待第一条心跳。'
  const recentStageEvents = useMemo(() => [...(task?.recent_logs ?? [])].slice(-4).reverse(), [task?.recent_logs])
  const draftSyncLabel = useMemo(() => {
    switch (draftSyncState) {
      case 'pending':
        return '草稿待保存'
      case 'saving':
        return '草稿保存中'
      case 'saved':
        return '草稿已自动保存'
      case 'error':
        return '草稿保存失败'
      default:
        return '草稿未同步'
    }
  }, [draftSyncState])

  function updatePreprocessConfig(updater: (current: PipelineConfig['preprocess']) => PipelineConfig['preprocess']) {
    setPipelineConfig((current) => ({ ...current, preprocess: updater(current.preprocess) }))
  }

  function updateFlowDirectionConfig(
    updater: (current: PipelineConfig['flow_direction']) => PipelineConfig['flow_direction'],
  ) {
    setPipelineConfig((current) => ({ ...current, flow_direction: updater(current.flow_direction) }))
  }

  function updateFlowAccumulationConfig(
    updater: (current: PipelineConfig['flow_accumulation']) => PipelineConfig['flow_accumulation'],
  ) {
    setPipelineConfig((current) => ({ ...current, flow_accumulation: updater(current.flow_accumulation) }))
  }

  function updateChannelExtractConfig(
    updater: (current: PipelineConfig['channel_extract']) => PipelineConfig['channel_extract'],
  ) {
    setPipelineConfig((current) => ({ ...current, channel_extract: updater(current.channel_extract) }))
  }

  function updateRuntimeConfig(updater: (current: PipelineConfig) => PipelineConfig) {
    setPipelineConfig((current) => updater(current))
  }

  function replaceTaskCollection(nextTasks: TaskSnapshot[]) {
    const ordered = [...nextTasks].sort((left, right) => Date.parse(right.updated_at) - Date.parse(left.updated_at))
    setTasks(ordered)
    setSelectedTaskId((current) => {
      if (current !== null && ordered.some((item) => item.task_id === current)) {
        return current
      }

      return ordered[0]?.task_id ?? null
    })
  }

  function applyTaskSnapshot(snapshot: TaskSnapshot) {
    setTasks((current) => {
      const next = [...current.filter((item) => item.task_id !== snapshot.task_id), snapshot]
      return next.sort((left, right) => Date.parse(right.updated_at) - Date.parse(left.updated_at))
    })
    setSelectedTaskId(snapshot.task_id)
    setLastBackendSyncAt(Date.now())
  }

  function replaceLibraryFile(
    kind: UploadedFileKind,
    file: UploadedFileInfo,
    previousPath?: string,
  ) {
    const setter = kind === 'input' ? setInputFiles : setMaskFiles
    setter((current) => {
      const next = current.filter((item) => item.stored_path !== (previousPath ?? file.stored_path))
      next.unshift(file)
      return next
    })
  }

  function applyDraftStateToForm(draftState: DraftTaskState | null) {
    isApplyingDraftStateRef.current = true
    try {
      setSelectedFile(null)
      setSelectedMaskFile(null)
      setUploadedInputFile(null)
      setUploadedMaskFile(null)
      setSelectedHistoryInputPath(draftState?.input_path ?? '')
      setSelectedHistoryMaskPath(draftState?.mask_path ?? '')
      setOutputFileName(extractOutputFileName(draftState?.output_path))
      setPipelineConfig(
        draftState?.config
          ? structuredClone(draftState.config)
          : structuredClone(DEFAULT_PIPELINE_CONFIG),
      )
      setInheritIntermediates(draftState?.inherit_intermediates ?? true)
      lastDraftStateSignatureRef.current = buildDraftStateSignature(
        draftState ?? {
          output_path: buildOutputPath('example-channel-result.png'),
          config: structuredClone(DEFAULT_PIPELINE_CONFIG),
          inherit_intermediates: true,
          inherit_stage_outputs: [],
        },
      )
    } finally {
      window.setTimeout(() => {
        isApplyingDraftStateRef.current = false
      }, 0)
    }
  }

  function buildCurrentDraftState(): DraftTaskState {
    return {
      input_path: activeInputFile?.stored_path ?? null,
      mask_path: activeMaskFile?.stored_path ?? null,
      output_path: outputPath,
      config: structuredClone(pipelineConfig),
      inherit_intermediates: inheritIntermediates,
      inherit_stage_outputs: [],
    }
  }

  async function refreshAllData() {
    const [taskSnapshots, nextInputFiles, nextMaskFiles] = await Promise.all([
      listTasks(),
      listUploadedFiles('input'),
      listUploadedFiles('mask'),
    ])
    replaceTaskCollection(taskSnapshots)
    setInputFiles(nextInputFiles)
    setMaskFiles(nextMaskFiles)
    setLastBackendSyncAt(Date.now())
  }

  useEffect(() => {
    void (async () => {
      try {
        await refreshAllData()
        const storedTaskId = window.localStorage.getItem(ACTIVE_TASK_STORAGE_KEY)
        if (storedTaskId !== null) {
          setSelectedTaskId(storedTaskId)
        }
      } catch (error) {
        setErrorMessage((error as Error).message)
      }
    })()
  }, [])

  useEffect(() => {
    if (selectedTaskId === null) {
      window.localStorage.removeItem(ACTIVE_TASK_STORAGE_KEY)
      return
    }

    window.localStorage.setItem(ACTIVE_TASK_STORAGE_KEY, selectedTaskId)
  }, [selectedTaskId])

  useEffect(() => {
    if (task === null) {
      setDraftSyncState('idle')
      setDraftSyncError(null)
      setLastDraftSavedAt(null)
      return
    }

    if (task.status === 'draft') {
      applyDraftStateToForm(task.draft_state)
      setDraftSyncState('saved')
      setDraftSyncError(null)
      setLastDraftSavedAt(parseTimestampMs(task.updated_at) ?? Date.now())
      return
    }

    setDraftSyncState('idle')
    setDraftSyncError(null)
    setLastDraftSavedAt(null)
  }, [task?.task_id, task?.status, task?.draft_state, task?.updated_at])

  useEffect(() => {
    return () => {
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current)
      }
      if (draftSaveTimerRef.current !== null) {
        window.clearTimeout(draftSaveTimerRef.current)
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

    return () => URL.revokeObjectURL(objectUrl)
  }, [selectedFile])

  useEffect(() => {
    if (selectedMaskFile === null) {
      setLocalMaskPreviewUrl((current) => {
        if (current !== null) {
          URL.revokeObjectURL(current)
        }
        return null
      })
      return
    }

    const objectUrl = URL.createObjectURL(selectedMaskFile)
    setLocalMaskPreviewUrl((current) => {
      if (current !== null) {
        URL.revokeObjectURL(current)
      }
      return objectUrl
    })

    return () => URL.revokeObjectURL(objectUrl)
  }, [selectedMaskFile])

  useEffect(() => {
    if (task === null || task.status !== 'draft') {
      return
    }

    if (isApplyingDraftStateRef.current) {
      return
    }

    const nextDraftState = buildCurrentDraftState()
    const nextSignature = buildDraftStateSignature(nextDraftState)
    if (nextSignature === lastDraftStateSignatureRef.current) {
      return
    }

    setDraftSyncState('pending')
    setDraftSyncError(null)

    if (draftSaveTimerRef.current !== null) {
      window.clearTimeout(draftSaveTimerRef.current)
    }

    draftSaveTimerRef.current = window.setTimeout(async () => {
      try {
        setDraftSyncState('saving')
        const snapshot = await updateDraftTaskState(task.task_id, nextDraftState)
        lastDraftStateSignatureRef.current = nextSignature
        setDraftSyncState('saved')
        setDraftSyncError(null)
        setLastDraftSavedAt(parseTimestampMs(snapshot.updated_at) ?? Date.now())
        setTasks((current) =>
          current
            .map((item) => (item.task_id === snapshot.task_id ? snapshot : item))
            .sort((left, right) => Date.parse(right.updated_at) - Date.parse(left.updated_at)),
        )
        setLastBackendSyncAt(Date.now())
      } catch (error) {
        setDraftSyncState('error')
        setDraftSyncError((error as Error).message)
      } finally {
        draftSaveTimerRef.current = null
      }
    }, 400)

    return () => {
      if (draftSaveTimerRef.current !== null) {
        window.clearTimeout(draftSaveTimerRef.current)
      }
    }
  }, [
    activeInputFile?.stored_path,
    activeMaskFile?.stored_path,
    inheritIntermediates,
    outputPath,
    pipelineConfig,
    task?.task_id,
    task?.status,
  ])

  useEffect(() => {
    if (task === null || !['queued', 'running', 'pausing'].includes(task.status)) {
      return
    }

    timerRef.current = window.setTimeout(async () => {
      if (pollInFlightRef.current) {
        return
      }

      pollInFlightRef.current = true
      try {
        const [snapshot, taskSnapshots] = await Promise.all([getTask(task.task_id), listTasks()])
        setLastBackendSyncAt(Date.now())
        replaceTaskCollection(taskSnapshots.map((item) => (item.task_id === snapshot.task_id ? snapshot : item)))
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

  function setInheritedStageOutputsForTask(nextStages: PipelineStage[]) {
    if (task === null) {
      return
    }

    const normalizedStages = normalizeInheritedStageSelection(nextStages, completedStageSet)
    setInheritStageOutputsByTaskId((current) => ({
      ...current,
      [task.task_id]: normalizedStages,
    }))
  }

  function handleToggleInheritedStage(stage: PipelineStage) {
    if (task === null || !completedStageSet.has(stage)) {
      return
    }

    const targetIndex = PIPELINE_STAGE_ORDER.indexOf(stage)
    const isEnabled = inheritedStageSet.has(stage)
    const nextStages = PIPELINE_STAGE_ORDER.filter((candidateStage) => {
      if (!completedStageSet.has(candidateStage)) {
        return false
      }

      const candidateIndex = PIPELINE_STAGE_ORDER.indexOf(candidateStage)
      return isEnabled ? candidateIndex < targetIndex : candidateIndex <= targetIndex
    })

    setInheritedStageOutputsForTask(nextStages)
  }

  async function uploadChosenFile(file: File, kind: UploadedFileKind) {
    const uploadTokenRef = kind === 'input' ? inputUploadTokenRef : maskUploadTokenRef
    const uploadToken = uploadTokenRef.current + 1
    uploadTokenRef.current = uploadToken

    if (kind === 'input') {
      setIsUploadingInput(true)
    } else {
      setIsUploadingMask(true)
    }

    setErrorMessage(null)
    try {
      const result = await uploadInputFile(file, kind)
      if (uploadTokenRef.current !== uploadToken) {
        return
      }

      replaceLibraryFile(kind, result)
      if (kind === 'input') {
        setUploadedInputFile(result)
      } else {
        setUploadedMaskFile(result)
        updatePreprocessConfig((current) => ({ ...current, use_mask: true }))
      }
    } catch (error) {
      if (uploadTokenRef.current === uploadToken) {
        setErrorMessage((error as Error).message)
      }
    } finally {
      if (uploadTokenRef.current === uploadToken) {
        if (kind === 'input') {
          setIsUploadingInput(false)
        } else {
          setIsUploadingMask(false)
        }
      }
    }
  }

  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null
    setSelectedFile(file)
    setUploadedInputFile(null)
    setSelectedHistoryInputPath('')
    setErrorMessage(null)
    if (file !== null) {
      void uploadChosenFile(file, 'input')
    }
  }

  function handleMaskFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null
    setSelectedMaskFile(file)
    setUploadedMaskFile(null)
    setSelectedHistoryMaskPath('')
    setErrorMessage(null)
    updatePreprocessConfig((current) => ({ ...current, use_mask: file !== null || current.use_mask }))
    if (file !== null) {
      void uploadChosenFile(file, 'mask')
    }
  }

  function handleHistoryFileSelect(kind: UploadedFileKind, path: string) {
    setErrorMessage(null)
    if (kind === 'input') {
      setSelectedHistoryInputPath(path)
      setSelectedFile(null)
      setUploadedInputFile(null)
    } else {
      setSelectedHistoryMaskPath(path)
      setSelectedMaskFile(null)
      setUploadedMaskFile(null)
      updatePreprocessConfig((current) => ({ ...current, use_mask: path !== '' || current.use_mask }))
    }
  }

  async function handleRenameManagedFile(kind: UploadedFileKind, file: UploadedFileInfo) {
    const nextName = window.prompt('输入新的文件名', file.filename)
    if (nextName === null || nextName.trim() === '' || nextName.trim() === file.filename) {
      return
    }

    setErrorMessage(null)
    try {
      const renamed = await renameUploadedFile({
        kind,
        stored_path: file.stored_path,
        name: nextName.trim(),
      })
      replaceLibraryFile(kind, renamed, file.stored_path)
      if (kind === 'input') {
        if (selectedHistoryInputPath === file.stored_path) {
          setSelectedHistoryInputPath(renamed.stored_path)
        }
        if (uploadedInputFile?.stored_path === file.stored_path) {
          setUploadedInputFile(renamed)
        }
      } else {
        if (selectedHistoryMaskPath === file.stored_path) {
          setSelectedHistoryMaskPath(renamed.stored_path)
        }
        if (uploadedMaskFile?.stored_path === file.stored_path) {
          setUploadedMaskFile(renamed)
        }
      }
    } catch (error) {
      setErrorMessage((error as Error).message)
    }
  }

  async function handleDeleteManagedFile(kind: UploadedFileKind, file: UploadedFileInfo) {
    if (!window.confirm(`确认删除 ${file.filename} 吗？`)) {
      return
    }

    setErrorMessage(null)
    try {
      await deleteUploadedFile({ kind, stored_path: file.stored_path })
      if (kind === 'input') {
        setInputFiles((current) => current.filter((item) => item.stored_path !== file.stored_path))
        if (selectedHistoryInputPath === file.stored_path) {
          setSelectedHistoryInputPath('')
        }
        if (uploadedInputFile?.stored_path === file.stored_path) {
          setUploadedInputFile(null)
        }
      } else {
        setMaskFiles((current) => current.filter((item) => item.stored_path !== file.stored_path))
        if (selectedHistoryMaskPath === file.stored_path) {
          setSelectedHistoryMaskPath('')
        }
        if (uploadedMaskFile?.stored_path === file.stored_path) {
          setUploadedMaskFile(null)
        }
      }
    } catch (error) {
      setErrorMessage((error as Error).message)
    }
  }

  function buildTaskPayload(endStage?: PipelineStage): CreateTaskRequest | null {
    if (activeInputFile === null) {
      setErrorMessage('请先为当前草稿选择输入主图。')
      return null
    }

    if (pipelineConfig.preprocess.use_mask && activeMaskFile === null) {
      setErrorMessage('当前已开启用户遮罩，但还没有可用的遮罩文件。')
      return null
    }

    return {
      input_path: activeInputFile.stored_path,
      mask_path: pipelineConfig.preprocess.use_mask && activeMaskFile !== null ? activeMaskFile.stored_path : null,
      output_path: outputPath,
      config: pipelineConfig,
      end_stage: endStage ?? 'channel_extract',
      inherit_intermediates: inheritIntermediates,
      inherit_stage_outputs: [],
    }
  }

  async function handleCreateDraftTask() {
    const defaultName = `任务 ${tasks.length + 1}`
    const requestedName = window.prompt('输入任务名称', defaultName)
    if (requestedName === null) {
      return
    }

    setErrorMessage(null)
    setIsTaskActionPending(true)
    try {
      const snapshot = await createDraftTask({ name: requestedName.trim() || defaultName })
      applyTaskSnapshot(snapshot)
    } catch (error) {
      setErrorMessage((error as Error).message)
    } finally {
      setIsTaskActionPending(false)
    }
  }

  async function handleStartDraftTask(endStage?: PipelineStage) {
    if (task === null || task.status !== 'draft') {
      setErrorMessage('请先在右侧新建并选中一个草稿任务。')
      return
    }

    const payload = buildTaskPayload(endStage)
    if (payload === null) {
      return
    }

    setErrorMessage(null)
    setIsSubmitting(true)
    try {
      const snapshot = await startExistingTask(task.task_id, payload)
      applyTaskSnapshot(snapshot)
    } catch (error) {
      setErrorMessage((error as Error).message)
    } finally {
      setIsSubmitting(false)
    }
  }

  async function handlePauseTask() {
    if (task === null) {
      return
    }

    setErrorMessage(null)
    setIsTaskActionPending(true)
    try {
      applyTaskSnapshot(await pauseTask(task.task_id))
    } catch (error) {
      setErrorMessage((error as Error).message)
    } finally {
      setIsTaskActionPending(false)
    }
  }

  async function handleResumeTask() {
    if (task === null) {
      return
    }

    setErrorMessage(null)
    setIsTaskActionPending(true)
    try {
      applyTaskSnapshot(await resumeTask(task.task_id))
    } catch (error) {
      setErrorMessage((error as Error).message)
    } finally {
      setIsTaskActionPending(false)
    }
  }

  async function handleContinueTask(endStage: PipelineStage = 'channel_extract') {
    if (task === null) {
      setErrorMessage('请先选择一个任务。')
      return
    }

    if (task.status === 'draft') {
      await handleStartDraftTask(endStage)
      return
    }

    setErrorMessage(null)
    setIsTaskActionPending(true)
    try {
      applyTaskSnapshot(
        await continueTask(task.task_id, {
          end_stage: endStage,
          inherit_intermediates: inheritIntermediates,
          inherit_stage_outputs: inheritStageOutputs,
        }),
      )
    } catch (error) {
      setErrorMessage((error as Error).message)
    } finally {
      setIsTaskActionPending(false)
    }
  }

  async function handleRerunTask() {
    if (task === null) {
      return
    }

    setErrorMessage(null)
    setIsTaskActionPending(true)
    try {
      applyTaskSnapshot(await rerunTask(task.task_id))
    } catch (error) {
      setErrorMessage((error as Error).message)
    } finally {
      setIsTaskActionPending(false)
    }
  }

  async function handleCancelTask() {
    if (task === null) {
      return
    }

    setErrorMessage(null)
    setIsTaskActionPending(true)
    try {
      applyTaskSnapshot(await cancelTask(task.task_id))
    } catch (error) {
      setErrorMessage((error as Error).message)
    } finally {
      setIsTaskActionPending(false)
    }
  }

  async function handleRenameTask() {
    if (task === null) {
      return
    }

    const nextName = window.prompt('输入新的任务名称', task.name)
    if (nextName === null || nextName.trim() === '' || nextName.trim() === task.name) {
      return
    }

    setErrorMessage(null)
    setIsTaskActionPending(true)
    try {
      applyTaskSnapshot(await renameTask(task.task_id, { name: nextName.trim() }))
    } catch (error) {
      setErrorMessage((error as Error).message)
    } finally {
      setIsTaskActionPending(false)
    }
  }

  async function handleDeleteTask() {
    if (task === null || !window.confirm(`确认删除任务“${task.name}”吗？`)) {
      return
    }

    setErrorMessage(null)
    setIsTaskActionPending(true)
    try {
      await deleteTask(task.task_id)
      setTasks((current) => current.filter((item) => item.task_id !== task.task_id))
      setSelectedTaskId((current) => (current === task.task_id ? null : current))
    } catch (error) {
      setErrorMessage((error as Error).message)
    } finally {
      setIsTaskActionPending(false)
    }
  }

  const viewerResult = useMemo<ArtifactViewerResult | null>(() => {
    if (task?.result) {
      return task.result
    }

    return buildPreviewResult(
      localPreviewUrl ?? activeInputFile?.stored_path ?? null,
      localMaskPreviewUrl ?? activeMaskFile?.stored_path ?? null,
      task?.status === 'draft' ? `${task.name}（草稿预览）` : '未启动任务预览',
    )
  }, [activeInputFile, activeMaskFile, localMaskPreviewUrl, localPreviewUrl, task])

  const thresholdValue = pipelineConfig.channel_extract.accumulation_threshold
  const lengthThresholdValue = pipelineConfig.channel_extract.channel_length_threshold

  return (
    <main className="app-shell cockpit-shell">
      <section className="top-status-bar">
        <div className="top-status-title">
          <p className="eyebrow">河道提取 MVP</p>
          <h1>大尺寸栅格河道工作台</h1>
        </div>
        <div className="top-status-chips">
          <span>任务数 {tasks.length}</span>
          <span>当前任务 {task?.name ?? '未选中'}</span>
          <span>状态 {task?.status ?? '空闲'}</span>
          <span>阶段 {task?.progress.stage ?? '等待中'}</span>
          <span>进度 {progressPercent.toFixed(1)}%</span>
          <span>阈值 {thresholdValue}</span>
          <span>长度阈值 {lengthThresholdValue}</span>
          <span>自动遮罩 {pipelineConfig.preprocess.use_auto_mask ? '开' : '关'}</span>
          <span>用户遮罩 {pipelineConfig.preprocess.use_mask ? '开' : '关'}</span>
          <span>心跳 <RelativeTimeText timestampMs={heartbeatTimestampMs} /></span>
        </div>
      </section>

      <section className="workspace-shell">
        <aside className={`side-rail left-rail${leftRailCollapsed ? ' collapsed' : ''}`}>
          <div className="rail-header">
            <div>
              <h2>配置</h2>
              {!leftRailCollapsed ? <p className="micro-note">左侧只对草稿任务开放；运行中会自动锁定。</p> : null}
            </div>
            <button
              type="button"
              className="secondary-button compact-button"
              onClick={() => setLeftRailCollapsed((current) => !current)}
            >
              {leftRailCollapsed ? '展开' : '折叠'}
            </button>
          </div>

          {!leftRailCollapsed ? (
            <article className="panel control-panel rail-panel">
              <div className="panel-title-row">
                <h2>任务配置</h2>
                <div className="button-row inline-actions control-toolbar">
                  <button
                    className="secondary-button compact-button"
                    type="button"
                    onClick={() => void refreshAllData()}
                    disabled={isSubmitting || isTaskActionPending}
                  >
                    刷新
                  </button>
                  <button
                    className="primary-button compact-button"
                    type="button"
                    onClick={() => void handleStartDraftTask()}
                    disabled={leftControlsDisabled}
                  >
                    {task?.status === 'draft' ? '启动当前草稿' : '需先新建草稿'}
                  </button>
                </div>
              </div>

              {leftLockReason ? <p className="lock-banner">{leftLockReason}</p> : null}
              {task?.status === 'draft' ? (
                <div className={`draft-sync-banner ${draftSyncState}`}>
                  <strong>{draftSyncLabel}</strong>
                  <span>
                    {draftSyncState === 'error'
                      ? draftSyncError ?? '后端未接受这次草稿保存。'
                      : draftSyncState === 'saved' && lastDraftSavedAt !== null
                        ? <>最近一次 <RelativeTimeText timestampMs={lastDraftSavedAt} /></>
                        : draftSyncState === 'saving'
                          ? '正在把左侧配置写回后端。'
                          : draftSyncState === 'pending'
                            ? '检测到你刚刚修改了参数，稍后会自动保存。'
                            : '当前草稿会在你修改后自动写回后端。'}
                  </span>
                </div>
              ) : null}

              <fieldset className="control-lock-fieldset" disabled={leftControlsDisabled}>
                <div className="dense-form-grid compact-grid-v2">
                  <GroupHeader title="输入与遮罩" subtitle="选择文件后立即上传，并可管理现有输入与遮罩素材。" />

                  <div className="field compact-field">
                    <FieldLabel label="输入主图" hint="选择本地图像后会立即上传到输入库，并在查看区显示。" />
                    <input type="file" accept=".pgm,.png,.jpg,.jpeg,.bmp,.tif,.tiff" onChange={handleFileChange} />
                    <p className="micro-note">
                      {selectedFile
                        ? `${selectedFile.name} · ${formatFileSize(selectedFile.size)}${isUploadingInput ? ' · 上传中…' : ' · 已入库'}`
                        : '尚未选择输入主图。'}
                    </p>
                  </div>

                  <div className="field compact-field">
                    <FieldLabel label="当前输入来源" hint="当前草稿将使用的输入主图。" />
                    <p className="mono-copy micro-note">{activeInputFile?.stored_path ?? '未选择'}</p>
                  </div>

                  <ManagedFileList
                    title="输入素材库"
                    emptyText="当前还没有已上传输入。"
                    files={inputFiles}
                    selectedPath={selectedHistoryInputPath}
                    onSelect={(path) => handleHistoryFileSelect('input', path)}
                    onRename={(file) => void handleRenameManagedFile('input', file)}
                    onDelete={(file) => void handleDeleteManagedFile('input', file)}
                    disabled={leftControlsDisabled}
                  />

                  <div className="field compact-field">
                    <FieldLabel label="用户遮罩" hint="选择本地遮罩后会立即上传到遮罩库，并在查看区显示。" />
                    <input type="file" accept=".pgm,.png,.jpg,.jpeg,.bmp,.tif,.tiff" onChange={handleMaskFileChange} />
                    <p className="micro-note">
                      {selectedMaskFile
                        ? `${selectedMaskFile.name} · ${formatFileSize(selectedMaskFile.size)}${isUploadingMask ? ' · 上传中…' : ' · 已入库'}`
                        : '未提供用户遮罩。'}
                    </p>
                  </div>

                  <div className="field compact-field">
                    <FieldLabel label="当前遮罩来源" hint="当前草稿将使用的用户遮罩。" />
                    <p className="mono-copy micro-note">{activeMaskFile?.stored_path ?? '未选择'}</p>
                  </div>

                  <ManagedFileList
                    title="遮罩素材库"
                    emptyText="当前还没有已上传遮罩。"
                    files={maskFiles}
                    selectedPath={selectedHistoryMaskPath}
                    onSelect={(path) => handleHistoryFileSelect('mask', path)}
                    onRename={(file) => void handleRenameManagedFile('mask', file)}
                    onDelete={(file) => void handleDeleteManagedFile('mask', file)}
                    disabled={leftControlsDisabled}
                  />

                  <div className="field compact-field">
                    <FieldLabel label="输出文件名" hint="最终结果会写入 data/output 目录。" />
                    <input value={outputFileName} onChange={(event) => setOutputFileName(event.target.value)} />
                  </div>

                  <div className="field compact-field">
                    <FieldLabel label="输出路径" hint="当前草稿最终输出的目标位置。" />
                    <p className="mono-copy micro-note">{outputPath}</p>
                  </div>

                  <GroupHeader title="预处理" subtitle="高低映射、填洼、自动遮罩与用户遮罩参与方式。" />

                  <div className="field compact-field field-span-2">
                    <FieldLabel label="高低映射" hint="决定灰度和地形高低的映射关系。" />
                    <SegmentedToggleGroup
                      value={pipelineConfig.preprocess.height_mapping}
                      options={[
                        { value: 'bright_is_high', label: '亮高', title: '亮像素视为高地形' },
                        { value: 'dark_is_high', label: '暗高', title: '暗像素视为高地形' },
                      ]}
                      onChange={(value) => updatePreprocessConfig((current) => ({ ...current, height_mapping: value }))}
                      disabled={leftControlsDisabled}
                    />
                  </div>

                  <div className="field compact-field">
                    <FieldLabel label="平滑" hint="轻度平滑输入高程，减少单像素噪声。" />
                    <ToggleStateButton
                      label="平滑"
                      enabled={pipelineConfig.preprocess.smooth}
                      title="平滑"
                      onToggle={() => updatePreprocessConfig((current) => ({ ...current, smooth: !current.smooth }))}
                      disabled={leftControlsDisabled}
                    />
                  </div>

                  <div className="field compact-field">
                    <FieldLabel label="填洼修复" hint="修复封闭洼地和无出口平坡。" />
                    <ToggleStateButton
                      label="填洼"
                      enabled={pipelineConfig.preprocess.fill_sinks}
                      title="填洼"
                      onToggle={() => updatePreprocessConfig((current) => ({ ...current, fill_sinks: !current.fill_sinks }))}
                      disabled={leftControlsDisabled}
                    />
                  </div>

                  <div className="field compact-field">
                    <FieldLabel label="自动遮罩" hint="自动识别有效区并直接参与后续计算。" />
                    <ToggleStateButton
                      label="自动遮罩"
                      enabled={pipelineConfig.preprocess.use_auto_mask}
                      title="自动遮罩"
                      onToggle={() =>
                        updatePreprocessConfig((current) => ({ ...current, use_auto_mask: !current.use_auto_mask }))
                      }
                      disabled={leftControlsDisabled}
                    />
                  </div>

                  <div className="field compact-field">
                    <FieldLabel label="用户遮罩" hint="启用后把当前遮罩作为额外约束，只在遮罩允许区域内计算。" />
                    <ToggleStateButton
                      label="用户遮罩"
                      enabled={pipelineConfig.preprocess.use_mask}
                      title="用户遮罩"
                      onToggle={() => updatePreprocessConfig((current) => ({ ...current, use_mask: !current.use_mask }))}
                      disabled={leftControlsDisabled || activeMaskFile === null}
                      disabledLabel="需先选遮罩"
                    />
                  </div>

                  <div className="field compact-field">
                    <FieldLabel label="平滑核" hint="越大越平滑，但细节更容易丢失。" />
                    <input
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

                  <div className="field compact-field">
                    <FieldLabel label="NoData" hint="指定无效像素值，留空表示不使用。" />
                    <input
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

                  <div className="field compact-field">
                    <FieldLabel label="边界敏感度" hint="越高越容易把边缘背景识别成无效区域。" />
                    <input
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

                  <div className="field compact-field">
                    <FieldLabel label="纹理敏感度" hint="越高越容易把低纹理区域识别为背景。" />
                    <input
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

                  <div className="field compact-field">
                    <FieldLabel label="最小区域面积" hint="自动遮罩保留的最小有效区域面积。" />
                    <input
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

                  <GroupHeader title="流向与输出" subtitle="调节 D8 权重、汇流预览以及最终河道过滤。" />

                  <div className="field compact-field">
                    <FieldLabel label="Rust D8" hint="启用后优先使用 Rust 严格 D8 内核。" />
                    <ToggleStateButton
                      label="Rust D8"
                      enabled={pipelineConfig.flow_direction.use_rust_kernel}
                      title="Rust D8"
                      onToggle={() =>
                        updateFlowDirectionConfig((current) => ({ ...current, use_rust_kernel: !current.use_rust_kernel }))
                      }
                      disabled={leftControlsDisabled}
                    />
                  </div>

                  <div className="field compact-field">
                    <FieldLabel label="坡降权重" hint="严格下降时对最陡方向的偏好强度。" />
                    <input
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

                  <div className="field compact-field">
                    <FieldLabel label="平坡逃逸权重" hint="平坡导流时，偏向更快离开平坡区域。" />
                    <input
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

                  <div className="field compact-field">
                    <FieldLabel label="出口接近权重" hint="平坡导流时，偏向更接近强出口的方向。" />
                    <input
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

                  <div className="field compact-field">
                    <FieldLabel label="连续性权重" hint="平坡导流时，偏向更连续的整体方向。" />
                    <input
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

                  <div className="field compact-field">
                    <FieldLabel label="出口长度权重" hint="出口段越长时权重越高。" />
                    <input
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

                  <div className="field compact-field">
                    <FieldLabel label="出口距离权重" hint="控制平坡内部按距离归属出口段的强度。" />
                    <input
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

                  <div className="field compact-field">
                    <FieldLabel label="汇流预览增强" hint="开启后对汇流预览做对数增强。" />
                    <ToggleStateButton
                      label="对数增强"
                      enabled={pipelineConfig.flow_accumulation.normalize}
                      title="对数增强"
                      onToggle={() =>
                        updateFlowAccumulationConfig((current) => ({ ...current, normalize: !current.normalize }))
                      }
                      disabled={leftControlsDisabled}
                    />
                  </div>

                  <div className="field compact-field">
                    <FieldLabel label="河道阈值" hint="越低河道越密，越高越偏主干。" />
                    <input
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

                  <div className="field compact-field">
                    <FieldLabel label="河道长度阈值" hint="最终阶段低于该长度的河道会被判为非法，不显示。" />
                    <input
                      type="number"
                      min={1}
                      step={1}
                      value={lengthThresholdValue}
                      onChange={(event) =>
                        updateChannelExtractConfig((current) => ({
                          ...current,
                          channel_length_threshold: Math.max(1, Number(event.target.value) || 1),
                        }))
                      }
                    />
                  </div>

                  <div className="field compact-field">
                    <FieldLabel label="保留中间产物" hint="保留每个阶段的中间结果，便于查看和调试。" />
                    <ToggleStateButton
                      label="中间产物"
                      enabled={pipelineConfig.save_intermediates}
                      title="保留中间产物"
                      onToggle={() => updateRuntimeConfig((current) => ({ ...current, save_intermediates: !current.save_intermediates }))}
                      disabled={leftControlsDisabled}
                    />
                  </div>

                  <div className="field compact-field">
                    <FieldLabel label="继承中间产物" hint="继续运行时是否复用当前任务已完成阶段的产物。" />
                    <ToggleStateButton
                      label="阶段继承"
                      enabled={inheritIntermediates}
                      title="阶段继承"
                      onToggle={() => setInheritIntermediates((current) => !current)}
                      disabled={task === null || task.status === 'draft'}
                      disabledLabel="仅历史任务"
                    />
                  </div>

                  <div className="field compact-field">
                    <FieldLabel label="进度分片" hint="任务进度分片数。越大反馈越细。" />
                    <input
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

                  <div className="field compact-field field-span-2">
                    <GroupHeader title="阶段复用" subtitle="按阶段决定继续运行时复用哪些已有结果。" />
                    <p className="micro-note">
                      {task === null || task.status === 'draft'
                        ? '选择一个已完成的历史任务后，才能配置阶段复用。'
                        : inheritIntermediates
                          ? `当前会复用：${inheritedStageSummary}`
                          : '当前已关闭阶段继承。'}
                    </p>
                    <div className="inherit-stage-grid">
                      {STEP_ENTRIES.map((step) => {
                        const stageReady = completedStageSet.has(step.key)
                        return (
                          <ToggleStateButton
                            key={`inherit-${step.key}`}
                            label={step.title}
                            enabled={inheritIntermediates && inheritedStageSet.has(step.key)}
                            title={step.description}
                            onToggle={() => handleToggleInheritedStage(step.key)}
                            disabled={task === null || task.status === 'draft' || !stageReady}
                            disabledLabel={task === null || task.status === 'draft' ? '需选历史任务' : '暂无产物'}
                          />
                        )
                      })}
                    </div>
                  </div>
                </div>
              </fieldset>

              {errorMessage ? <p className="error-banner">{errorMessage}</p> : null}
            </article>
          ) : (
            <article className="panel collapsed-rail-panel">
              <span className="rail-kpi">{leftLockReason ? '已锁定' : '草稿可编辑'}</span>
            </article>
          )}
        </aside>

        <section className="center-stage">
          <MemoArtifactViewerTabs result={viewerResult} />
        </section>

        <aside className={`side-rail right-rail${rightRailCollapsed ? ' collapsed' : ''}`}>
          <div className="rail-header">
            <div>
              <h2>任务中心</h2>
              {!rightRailCollapsed ? <p className="micro-note">新建、重命名、删除、暂停、继续与分阶段启动。</p> : null}
            </div>
            <button
              type="button"
              className="secondary-button compact-button"
              onClick={() => setRightRailCollapsed((current) => !current)}
            >
              {rightRailCollapsed ? '展开' : '折叠'}
            </button>
          </div>

          {!rightRailCollapsed ? (
            <div className="side-stack">
              <article className="panel compact-panel">
                <div className="panel-title-row">
                  <h2>任务列表</h2>
                  <div className="button-row inline-actions">
                    <button type="button" className="primary-button compact-button" onClick={() => void handleCreateDraftTask()} disabled={isTaskActionPending || isSubmitting || activeRunningTask !== null}>
                      新建任务
                    </button>
                    <button type="button" className="secondary-button compact-button" onClick={() => void refreshAllData()} disabled={isTaskActionPending || isSubmitting}>
                      刷新
                    </button>
                  </div>
                </div>

                <div className="button-row inline-actions">
                  <button type="button" className="secondary-button compact-button" onClick={() => void handleRenameTask()} disabled={task === null || isTaskActionPending}>
                    重命名
                  </button>
                  <button type="button" className="secondary-button compact-button" onClick={() => void handleDeleteTask()} disabled={task === null || isTaskActionPending || isTaskActive}>
                    删除
                  </button>
                </div>

                <div className="task-list">
                  {tasks.length > 0 ? (
                    tasks.map((item) => {
                      const selected = item.task_id === task?.task_id
                      const itemHeartbeatTimestampMs = parseTimestampMs(item.progress.last_heartbeat_at)
                      return (
                        <button
                          key={item.task_id}
                          type="button"
                          className={`task-list-item${selected ? ' selected' : ''}`}
                          onClick={() => setSelectedTaskId(item.task_id)}
                        >
                          <span className="task-list-title">
                            {item.name}
                            <small>{item.status}</small>
                          </span>
                          <span className="task-list-meta">阶段 {item.progress.stage}</span>
                          <span className="task-list-meta">心跳 <RelativeTimeText timestampMs={itemHeartbeatTimestampMs} /></span>
                        </button>
                      )
                    })
                  ) : (
                    <p className="micro-note">还没有任务。请先在这里新建一个任务草稿。</p>
                  )}
                </div>
              </article>

              <article className="panel compact-panel">
                <h2>任务状态</h2>
                <div className="status-strip-grid">
                  <div className="metric-card compact-metric">
                    <span className="field-label">名称</span>
                    <strong>{task?.name ?? '未选任务'}</strong>
                  </div>
                  <div className="metric-card compact-metric">
                    <span className="field-label">状态</span>
                    <strong>{task?.status ?? '空闲'}</strong>
                  </div>
                  <div className="metric-card compact-metric">
                    <span className="field-label">阶段</span>
                    <strong>{task?.progress.stage ?? '等待中'}</strong>
                  </div>
                  <div className="metric-card compact-metric">
                    <span className="field-label">活跃度</span>
                    <strong><HeartbeatStateText status={task?.status ?? null} timestampMs={heartbeatTimestampMs} /></strong>
                  </div>
                </div>

                <div className="button-row">
                  <button type="button" className="secondary-button compact-button" onClick={() => void handlePauseTask()} disabled={!canPauseTask || isTaskActionPending}>
                    {task?.status === 'queued' ? '排队暂停' : '暂停'}
                  </button>
                  <button type="button" className="secondary-button compact-button" onClick={() => void handleResumeTask()} disabled={!canResumeTask || isTaskActionPending}>
                    继续
                  </button>
                  <button type="button" className="secondary-button compact-button" onClick={() => void handleContinueTask()} disabled={!(canContinueSelectedTask || task?.status === 'draft') || isTaskActionPending || leftControlsDisabled && task?.status === 'draft'}>
                    {task?.status === 'draft' ? '启动到最终阶段' : '继续到最终阶段'}
                  </button>
                  <button type="button" className="secondary-button compact-button" onClick={() => void handleRerunTask()} disabled={!canRerunSelectedTask || isTaskActionPending}>
                    重跑
                  </button>
                  <button type="button" className="secondary-button compact-button" onClick={() => void handleCancelTask()} disabled={!isTaskActive || isTaskActionPending}>
                    取消
                  </button>
                </div>

                <div className="task-activity-strip">
                  <span className="field-label">当前子步骤</span>
                  <strong>{task?.progress.message ?? '等待任务启动。'}</strong>
                </div>
                <div className="task-activity-strip">
                  <span className="field-label">最近活动</span>
                  <strong>{lastHeartbeatMessage}</strong>
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
                    {task?.progress.processed_units ?? 0}/{task?.progress.total_units ?? 0} 单元，最近心跳 <RelativeTimeText timestampMs={heartbeatTimestampMs} />
                  </p>
                  <p className="muted-copy">
                    后端快照 <RelativeTimeText timestampMs={backendUpdatedAtMs} />，本地同步 <RelativeTimeText timestampMs={lastBackendSyncAt} />
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
                <h2>阶段入口</h2>
                <p className="micro-note">草稿任务会从头启动到目标阶段；历史任务会按当前阶段复用策略继续运行。</p>
                <div className="step-entry-list dense-step-list">
                  {STEP_ENTRIES.map((step) => {
                    const stageCompleted = completedStageSet.has(step.key)
                    const stageInherited = inheritIntermediates && inheritedStageSet.has(step.key)
                    const stageButtonLabel =
                      task?.status === 'draft'
                        ? '从此启动'
                        : stageCompleted
                          ? '已完成'
                          : canContinueSelectedTask
                            ? '继续到此'
                            : isTaskActive
                              ? '运行中'
                              : '待命'
                    return (
                      <div key={step.key} className={`step-entry-card dense-step-card${stageCompleted ? ' stage-complete' : ''}`}>
                        <div className="step-entry-copy">
                          <strong>{step.title}</strong>
                          <p className="micro-note">{step.description}</p>
                          <p className="micro-note">
                            {stageCompleted
                              ? stageInherited
                                ? '继续运行时将复用本阶段产物。'
                                : '继续运行时会从本阶段或更早位置重算。'
                              : '当前任务尚未产出本阶段结果。'}
                          </p>
                        </div>
                        <button
                          type="button"
                          className="secondary-button compact-button"
                          disabled={task === null || isTaskActionPending || isTaskActive || (task.status === 'draft' ? leftControlsDisabled : !canContinueSelectedTask && !stageCompleted)}
                          onClick={() => void handleContinueTask(step.key)}
                        >
                          {stageButtonLabel}
                        </button>
                      </div>
                    )
                  })}
                </div>
              </article>

              <article className="panel compact-panel">
                <h2>日志与产物</h2>
                <div className="compact-meta-list">
                  <p><strong>任务目录</strong><span className="mono-copy">{task?.result?.task_directory ?? '待生成'}</span></p>
                  <p><strong>元数据</strong><span className="mono-copy">{task?.result?.metadata_path ?? '待生成'}</span></p>
                  <p><strong>最终输出</strong><span className="mono-copy">{task?.result?.channel_mask ?? outputPath}</span></p>
                  <p><strong>主图来源</strong><span className="mono-copy">{activeInputFile?.stored_path ?? '未选择'}</span></p>
                  <p><strong>用户遮罩</strong><span className="mono-copy">{pipelineConfig.preprocess.use_mask ? activeMaskFile?.stored_path ?? '未启用或未选择' : '未启用'}</span></p>
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
              <span className="rail-kpi">{progressPercent.toFixed(0)}%</span>
            </article>
          )}
        </aside>
      </section>
    </main>
  )
}
