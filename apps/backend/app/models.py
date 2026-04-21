"""
Shared models for the backend API and the internal task runner.

These models are kept in one place to reduce coupling between modules. The API,
task runner, and pipeline all exchange typed data structures instead of
reaching into each other's implementation details.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Return a timezone-aware timestamp for all task records."""

    return datetime.now(timezone.utc)


def default_task_name() -> str:
    """Return a stable fallback display name for new task records."""

    return f"Task {utc_now().strftime('%Y-%m-%d %H:%M:%S')}"


class TaskStatus(str, Enum):
    """Stable task lifecycle states exposed to the frontend."""

    DRAFT = "draft"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class PipelineStage(str, Enum):
    """Named pipeline stages used by both progress reporting and UI display."""

    IO = "io"
    PREPROCESS = "preprocess"
    FLOW_DIRECTION = "flow_direction"
    FLOW_ACCUMULATION = "flow_accumulation"
    CHANNEL_EXTRACT = "channel_extract"


PIPELINE_STAGE_SEQUENCE: tuple[PipelineStage, ...] = (
    PipelineStage.IO,
    PipelineStage.PREPROCESS,
    PipelineStage.FLOW_DIRECTION,
    PipelineStage.FLOW_ACCUMULATION,
    PipelineStage.CHANNEL_EXTRACT,
)


def get_stage_index(stage: PipelineStage) -> int:
    """Return the stable execution index for a pipeline stage."""

    return PIPELINE_STAGE_SEQUENCE.index(stage)


def get_next_stage(stage: PipelineStage) -> PipelineStage | None:
    """Return the next stage in execution order, if one exists."""

    stage_index = get_stage_index(stage)
    if stage_index >= len(PIPELINE_STAGE_SEQUENCE) - 1:
        return None

    return PIPELINE_STAGE_SEQUENCE[stage_index + 1]


class PreprocessConfig(BaseModel):
    """
    Configuration for terrain preprocessing.

    The fields are intentionally small for v1, but stable enough that future
    work can add algorithm-specific options without changing the stage contract.
    """

    height_mapping: Literal["bright_is_high", "dark_is_high"] = "bright_is_high"
    smooth: bool = True
    smooth_kernel_size: int = Field(default=3, ge=1, le=15)
    fill_sinks: bool = True
    preserve_nodata: bool = True
    nodata_value: float | None = Field(default=None, ge=0.0, le=255.0)
    use_auto_mask: bool = False
    auto_mask_border_sensitivity: float = Field(default=1.0, ge=0.1, le=5.0)
    auto_mask_texture_sensitivity: float = Field(default=1.0, ge=0.1, le=5.0)
    auto_mask_min_region_size: int = Field(default=2048, ge=1, le=10_000_000)
    use_mask: bool = False


class FlowDirectionConfig(BaseModel):
    """Configuration for flow-direction calculation."""

    method: Literal["D8"] = "D8"
    use_rust_kernel: bool = False
    slope_weight: float = Field(default=1.0, ge=0.0, le=10.0)
    flat_escape_weight: float = Field(default=0.6, ge=0.0, le=10.0)
    outlet_proximity_weight: float = Field(default=0.4, ge=0.0, le=10.0)
    continuity_weight: float = Field(default=0.3, ge=0.0, le=10.0)
    flat_outlet_length_weight: float = Field(default=0.35, ge=0.0, le=10.0)
    flat_outlet_distance_weight: float = Field(default=1.5, ge=0.0, le=10.0)


class FlowAccumulationConfig(BaseModel):
    """Configuration for flow accumulation."""

    normalize: bool = True


class ChannelExtractConfig(BaseModel):
    """Configuration for converting accumulation into a binary channel mask."""

    accumulation_threshold: float = Field(default=200.0, gt=0)
    channel_length_threshold: int = Field(default=1, ge=1)


class PipelineConfig(BaseModel):
    """
    Top-level pipeline configuration.

    Each stage owns a nested configuration model to avoid a flat and
    tightly-coupled parameter list.
    """

    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    flow_direction: FlowDirectionConfig = Field(default_factory=FlowDirectionConfig)
    flow_accumulation: FlowAccumulationConfig = Field(default_factory=FlowAccumulationConfig)
    channel_extract: ChannelExtractConfig = Field(default_factory=ChannelExtractConfig)
    save_intermediates: bool = True
    total_tiles: int = Field(default=64, ge=4, le=4096)


class RiverTaskRequest(BaseModel):
    """
    Request payload for a new river extraction task.

    The input and output paths are placeholders for now. We keep them in the
    contract from day one because long-running raster work will eventually need
    explicit file ownership and output destinations.
    """

    input_path: str = Field(min_length=1)
    mask_path: str | None = None
    output_path: str = Field(min_length=1)
    config: PipelineConfig = Field(default_factory=PipelineConfig)
    start_stage: PipelineStage | None = None
    end_stage: PipelineStage | None = None
    inherit_intermediates: bool = True
    inherit_stage_outputs: list[PipelineStage] | None = None


class DraftTaskState(BaseModel):
    """Persisted editable state for one task draft before execution starts."""

    input_path: str | None = None
    mask_path: str | None = None
    output_path: str = Field(default="data/output/example-channel-result.png", min_length=1)
    config: PipelineConfig = Field(default_factory=PipelineConfig)
    inherit_intermediates: bool = True
    inherit_stage_outputs: list[PipelineStage] | None = None


class DraftTaskRequest(BaseModel):
    """Request payload for creating one editable task draft."""

    name: str | None = Field(default=None, min_length=1, max_length=120)


class RenameTaskRequest(BaseModel):
    """Request payload for renaming one persisted task record."""

    name: str = Field(min_length=1, max_length=120)


class UploadedFileKind(str, Enum):
    """Stable file categories used by the asset manager."""

    INPUT = "input"
    MASK = "mask"


class UploadedFileInfo(BaseModel):
    """Metadata returned for one reusable uploaded file."""

    kind: UploadedFileKind
    filename: str
    stored_path: str
    size_bytes: int


class RenameUploadedFileRequest(BaseModel):
    """Rename one managed uploaded file."""

    kind: UploadedFileKind
    stored_path: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=240)


class DeleteUploadedFileRequest(BaseModel):
    """Delete one managed uploaded file."""

    kind: UploadedFileKind
    stored_path: str = Field(min_length=1)


class ContinueTaskRequest(BaseModel):
    """Request payload for continuing one existing task to a target stage."""

    end_stage: PipelineStage | None = None
    inherit_intermediates: bool = True
    inherit_stage_outputs: list[PipelineStage] | None = None


class TaskProgress(BaseModel):
    """
    Fine-grained progress information for the current stage.

    `processed_units` and `total_units` are the main hooks that let the UI show
    movement during large-image processing instead of looking frozen.
    """

    stage: PipelineStage
    percent: float = Field(default=0.0, ge=0.0, le=100.0)
    processed_units: int = Field(default=0, ge=0)
    total_units: int = Field(default=1, ge=1)
    message: str = ""
    eta_seconds: float | None = Field(default=None, ge=0.0)
    last_heartbeat_at: datetime | None = None
    last_heartbeat_message: str = ""


class ArtifactStatus(str, Enum):
    """Lifecycle state for one saved task artifact."""

    PENDING = "pending"
    READY = "ready"


class ArtifactRecord(BaseModel):
    """
    Metadata for one artifact exposed to the frontend.

    Each artifact keeps both status and file path so the UI can render stable
    tabs even before a stage has generated its preview image.
    """

    key: str
    label: str
    stage: PipelineStage
    status: ArtifactStatus = ArtifactStatus.PENDING
    path: str | None = None
    preview_path: str | None = None
    previewable: bool = True
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)


class PipelineResult(BaseModel):
    """
    Internal and external summary of pipeline outputs.

    The explicit artifact fields are deliberate extension points. Future work
    can fill these with real file paths or metadata without breaking the API.
    """

    task_directory: str | None = None
    metadata_path: str | None = None
    input_preview: str | None = None
    auto_mask: str | None = None
    terrain_preprocessed: str | None = None
    flow_direction: str | None = None
    flow_accumulation: str | None = None
    channel_mask: str | None = None
    artifacts: dict[str, ArtifactRecord] = Field(default_factory=dict)


class RiverTaskSnapshot(BaseModel):
    """Serializable snapshot returned by the task API."""

    task_id: str
    name: str
    status: TaskStatus
    draft_state: DraftTaskState | None = None
    progress: TaskProgress
    created_at: datetime
    updated_at: datetime
    result: PipelineResult | None = None
    error: str | None = None
    recent_logs: list[str] = Field(default_factory=list)
    last_completed_stage: PipelineStage | None = None
    completed_stages: list[PipelineStage] = Field(default_factory=list)


class RiverTaskRecord(BaseModel):
    """
    Mutable in-memory task record used by the task runner.

    The record keeps the original request so future resumable execution can
    rehydrate task inputs without digging through other modules.
    """

    task_id: str = Field(default_factory=lambda: uuid4().hex)
    name: str = Field(default_factory=default_task_name)
    draft_state: DraftTaskState | None = Field(default_factory=DraftTaskState)
    request: RiverTaskRequest | None = None
    status: TaskStatus = TaskStatus.DRAFT
    progress: TaskProgress = Field(
        default_factory=lambda: TaskProgress(stage=PipelineStage.IO, total_units=1)
    )
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    result: PipelineResult | None = None
    error: str | None = None
    recent_logs: list[str] = Field(default_factory=list)
    last_completed_stage: PipelineStage | None = None
    completed_stages: list[PipelineStage] = Field(default_factory=list)
