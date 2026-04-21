export type TaskStatus = 'draft' | 'queued' | 'running' | 'pausing' | 'paused' | 'completed' | 'failed' | 'canceled'

export type PipelineStage =
  | 'io'
  | 'preprocess'
  | 'flow_direction'
  | 'flow_accumulation'
  | 'channel_extract'

export type TaskProgress = {
  stage: PipelineStage
  percent: number
  processed_units: number
  total_units: number
  message: string
  eta_seconds: number | null
  last_heartbeat_at: string | null
  last_heartbeat_message: string
}

export type ArtifactStatus = 'pending' | 'ready'

export type ArtifactRecord = {
  key: string
  label: string
  stage: PipelineStage
  status: ArtifactStatus
  path: string | null
  preview_path: string | null
  previewable: boolean
  width: number | null
  height: number | null
}

export type PipelineResult = {
  task_directory: string | null
  metadata_path: string | null
  input_preview: string | null
  auto_mask: string | null
  terrain_preprocessed: string | null
  flow_direction: string | null
  flow_accumulation: string | null
  channel_mask: string | null
  artifacts: Record<string, ArtifactRecord>
}

export type ArtifactViewerResult = PipelineResult

export type PipelineConfig = {
  preprocess: {
    height_mapping: 'bright_is_high' | 'dark_is_high'
    smooth: boolean
    smooth_kernel_size: number
    fill_sinks: boolean
    preserve_nodata: boolean
    nodata_value: number | null
    use_auto_mask: boolean
    auto_mask_border_sensitivity: number
    auto_mask_texture_sensitivity: number
    auto_mask_min_region_size: number
    use_mask: boolean
  }
  flow_direction: {
    method: 'D8'
    use_rust_kernel: boolean
    slope_weight: number
    flat_escape_weight: number
    outlet_proximity_weight: number
    continuity_weight: number
    flat_outlet_length_weight: number
    flat_outlet_distance_weight: number
  }
  flow_accumulation: {
    normalize: boolean
  }
  channel_extract: {
    accumulation_threshold: number
    channel_length_threshold: number
  }
  save_intermediates: boolean
  total_tiles: number
}

export type UploadedFileKind = 'input' | 'mask'

export type DraftTaskState = {
  input_path?: string | null
  mask_path?: string | null
  output_path: string
  config: PipelineConfig
  inherit_intermediates: boolean
  inherit_stage_outputs?: PipelineStage[] | null
}

export type CreateTaskRequest = {
  input_path: string
  mask_path?: string | null
  output_path: string
  config: PipelineConfig
  start_stage?: PipelineStage | null
  end_stage?: PipelineStage | null
  inherit_intermediates?: boolean
  inherit_stage_outputs?: PipelineStage[] | null
}

export type ContinueTaskRequest = {
  end_stage?: PipelineStage | null
  inherit_intermediates?: boolean
  inherit_stage_outputs?: PipelineStage[] | null
}

export type UploadedFileInfo = {
  kind: UploadedFileKind
  filename: string
  stored_path: string
  size_bytes: number
}

export type CreateDraftTaskRequest = {
  name?: string | null
}

export type RenameTaskRequest = {
  name: string
}

export type RenameUploadedFileRequest = {
  kind: UploadedFileKind
  stored_path: string
  name: string
}

export type DeleteUploadedFileRequest = {
  kind: UploadedFileKind
  stored_path: string
}

export type TaskSnapshot = {
  task_id: string
  name: string
  status: TaskStatus
  draft_state: DraftTaskState | null
  progress: TaskProgress
  created_at: string
  updated_at: string
  result: PipelineResult | null
  error: string | null
  recent_logs: string[]
  last_completed_stage: PipelineStage | null
  completed_stages: PipelineStage[]
}
