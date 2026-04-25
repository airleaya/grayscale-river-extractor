"""
Stage orchestration for the river extraction pipeline.

The backend is intentionally split into two layers:
1. This module coordinates stage ordering, progress semantics, and artifact IO.
2. `raster_algorithms.py` owns the pure raster transforms and preview generation.

Keeping those responsibilities separate gives us a cleaner path for later
performance work, algorithm replacement, and test coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from json import dumps
from pathlib import Path
from time import monotonic
from typing import Callable, Protocol

import numpy as np
from PIL import Image

from .logging_utils import get_logger
from .models import (
    ArtifactRecord,
    ArtifactStatus,
    PIPELINE_STAGE_SEQUENCE,
    PipelineConfig,
    PipelineResult,
    PipelineStage,
    RiverTaskRequest,
    get_stage_index,
)
from .raster_algorithms import (
    accumulation_preview_image,
    apply_box_smoothing,
    apply_height_mapping,
    auto_mask_preview_image,
    build_channel_mask,
    build_valid_mask,
    channel_preview_image,
    compute_d8_flow_directions,
    compute_flow_accumulation,
    direction_preview_image,
    fill_depressions_priority_flood,
    fill_depth_preview_image,
    fill_local_sinks,
    generate_auto_mask,
    load_height_array,
    load_mask_array,
    rust_priority_flood_available,
    terrain_preview_image,
    terrain_statistics_message,
    DEFAULT_FILL_SINK_MAX_ITERATIONS,
)


pipeline_logger = get_logger("river.pipeline")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_LABELS = {
    "input_preview": "输入图像",
    "user_mask": "用户遮罩",
    "auto_mask": "自动遮罩",
    "terrain_preprocessed": "预处理地形",
    "fill_depth": "填洼深度",
    "flow_direction": "流向",
    "flow_accumulation": "汇流累积",
    "channel_mask": "河道结果",
}


@dataclass(slots=True)
class PipelineContext:
    """Runtime paths and metadata shared across pipeline stages."""

    task_id: str
    request: RiverTaskRequest
    input_path: Path
    mask_path: Path | None
    output_path: Path
    task_directory: Path
    metadata_path: Path


def resolve_project_path(path_value: str) -> Path:
    """Resolve a task path against the project root unless it is already absolute."""

    path = Path(path_value)
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def ensure_parent_directory(path: Path) -> None:
    """Create the parent directory for an output file if it is missing."""

    path.parent.mkdir(parents=True, exist_ok=True)


def save_preview_image(image: Image.Image, path: Path, max_side: int) -> Image.Image:
    """Save a bounded preview image and return the saved image object."""

    ensure_parent_directory(path)
    preview_image = image
    longest_side = max(preview_image.size)
    if longest_side > max_side:
        preview_image = image.copy()
        preview_image.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
    preview_image.save(path)
    return preview_image


def to_project_relative_path(path: Path) -> str:
    """Convert an absolute project path into the stable API path format."""

    return path.relative_to(PROJECT_ROOT).as_posix()


def build_pipeline_context(task_id: str, request: RiverTaskRequest) -> PipelineContext:
    """Build resolved runtime paths for one pipeline execution."""

    input_path = resolve_project_path(request.input_path)
    mask_path = resolve_project_path(request.mask_path) if request.mask_path else None
    output_path = resolve_project_path(request.output_path)
    task_directory = PROJECT_ROOT / "data" / "output" / "tasks" / task_id
    task_directory.mkdir(parents=True, exist_ok=True)
    metadata_path = task_directory / "metadata.json"
    ensure_parent_directory(output_path)
    return PipelineContext(
        task_id=task_id,
        request=request,
        input_path=input_path,
        mask_path=mask_path,
        output_path=output_path,
        task_directory=task_directory,
        metadata_path=metadata_path,
    )


def build_default_pipeline_result(context: PipelineContext) -> PipelineResult:
    """Create a stable artifact registry before any stage has produced output."""

    artifacts = {
        "input_preview": ArtifactRecord(
            key="input_preview",
            label=ARTIFACT_LABELS["input_preview"],
            stage=PipelineStage.IO,
        ),
        "terrain_preprocessed": ArtifactRecord(
            key="terrain_preprocessed",
            label=ARTIFACT_LABELS["terrain_preprocessed"],
            stage=PipelineStage.PREPROCESS,
        ),
        "fill_depth": ArtifactRecord(
            key="fill_depth",
            label=ARTIFACT_LABELS["fill_depth"],
            stage=PipelineStage.PREPROCESS,
        ),
        "user_mask": ArtifactRecord(
            key="user_mask",
            label=ARTIFACT_LABELS["user_mask"],
            stage=PipelineStage.IO,
        ),
        "auto_mask": ArtifactRecord(
            key="auto_mask",
            label=ARTIFACT_LABELS["auto_mask"],
            stage=PipelineStage.PREPROCESS,
        ),
        "flow_direction": ArtifactRecord(
            key="flow_direction",
            label=ARTIFACT_LABELS["flow_direction"],
            stage=PipelineStage.FLOW_DIRECTION,
        ),
        "flow_accumulation": ArtifactRecord(
            key="flow_accumulation",
            label=ARTIFACT_LABELS["flow_accumulation"],
            stage=PipelineStage.FLOW_ACCUMULATION,
        ),
        "channel_mask": ArtifactRecord(
            key="channel_mask",
            label=ARTIFACT_LABELS["channel_mask"],
            stage=PipelineStage.CHANNEL_EXTRACT,
        ),
    }
    return PipelineResult(
        task_directory=to_project_relative_path(context.task_directory),
        metadata_path=to_project_relative_path(context.metadata_path),
        artifacts=artifacts,
    )


def prepare_initial_result(task_id: str, request: RiverTaskRequest) -> PipelineResult:
    """Prepare the empty artifact registry used before the first stage completes."""

    context = build_pipeline_context(task_id, request)
    result = build_default_pipeline_result(context)
    write_pipeline_metadata(result, context.metadata_path)
    return result


def sync_legacy_artifact_fields(result: PipelineResult) -> None:
    """Mirror artifact registry paths into the legacy explicit fields."""

    result.input_preview = result.artifacts["input_preview"].preview_path
    result.auto_mask = result.artifacts["auto_mask"].preview_path
    result.terrain_preprocessed = result.artifacts["terrain_preprocessed"].preview_path
    result.flow_direction = result.artifacts["flow_direction"].preview_path
    result.flow_accumulation = result.artifacts["flow_accumulation"].preview_path
    result.channel_mask = result.artifacts["channel_mask"].preview_path


def write_pipeline_metadata(result: PipelineResult, metadata_path: Path) -> None:
    """Persist task artifact metadata so finished and running tasks are inspectable."""

    ensure_parent_directory(metadata_path)
    metadata_path.write_text(
        dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def update_result_with_artifact(
    result: PipelineResult,
    artifact: ArtifactRecord,
    metadata_path: Path,
) -> PipelineResult:
    """Update one artifact entry and immediately refresh the task metadata file."""

    result.artifacts[artifact.key] = artifact
    sync_legacy_artifact_fields(result)
    write_pipeline_metadata(result, metadata_path)
    return result


def get_artifact_label(key: str) -> str:
    """Return the user-facing label for one artifact key."""

    return ARTIFACT_LABELS.get(key, key)


def _user_mask_data_path(context: PipelineContext) -> Path:
    """Return the persisted task-local user-mask raster path."""

    return context.task_directory / "user_mask.npy"


def _user_mask_preview_path(context: PipelineContext) -> Path:
    """Return the persisted task-local user-mask preview path."""

    return context.task_directory / "user_mask.png"


def load_user_mask_for_shape(
    context: PipelineContext,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    """Load one user mask from cache or source file and validate its raster size."""

    cached_path = _user_mask_data_path(context)
    if cached_path.exists():
        user_mask = np.load(cached_path).astype(bool)
    else:
        if context.mask_path is None:
            raise FileNotFoundError("当前任务未提供用户遮罩路径。")
        if not context.mask_path.exists():
            raise FileNotFoundError(f"用户遮罩栅格未找到：{context.mask_path}")
        with Image.open(context.mask_path) as image:
            user_mask = load_mask_array(image)

    if user_mask.shape != expected_shape:
        raise ValueError(
            "用户遮罩栅格尺寸与地形栅格不一致："
            f"{user_mask.shape} != {expected_shape}"
        )

    return user_mask.astype(bool, copy=False)


def persist_user_mask_artifact(
    context: PipelineContext,
    user_mask: np.ndarray,
    stage: PipelineStage,
) -> ArtifactRecord:
    """Persist one validated user mask and return the publishable artifact record."""

    user_mask_data_path = _user_mask_data_path(context)
    user_mask_preview_path = _user_mask_preview_path(context)
    np.save(user_mask_data_path, user_mask.astype(np.uint8))
    auto_mask_preview_image(user_mask).save(user_mask_preview_path)
    return ArtifactRecord(
        key="user_mask",
        label=get_artifact_label("user_mask"),
        stage=stage,
        status=ArtifactStatus.READY,
        path=to_project_relative_path(user_mask_data_path),
        preview_path=to_project_relative_path(user_mask_preview_path),
        width=int(user_mask.shape[1]),
        height=int(user_mask.shape[0]),
    )


class ProgressReporter(Protocol):
    """Interface used by pipeline stages to report work without knowing the UI."""

    def begin_stage(self, stage: PipelineStage, total_units: int, message: str) -> None:
        """Start a stage and reset unit-level progress."""

    def advance(self, units: int = 1, message: str = "") -> None:
        """Advance the current stage by a number of work units."""

    def complete_stage(self, message: str = "") -> None:
        """Mark the current stage as complete."""

    def log(self, message: str) -> None:
        """Append a human-readable task log entry."""

    def publish_artifact(self, artifact: ArtifactRecord) -> None:
        """Publish one saved artifact so the task snapshot can expose it immediately."""

    def heartbeat(self, message: str = "", force: bool = False) -> None:
        """Refresh task liveness without requiring a progress-unit increment."""

    def set_parallel_work(
        self,
        label: str,
        strategy: str,
        chunks: list[tuple[str, str, int]],
    ) -> None:
        """Publish one structured parallel-work snapshot for the current stage."""

    def update_parallel_chunk(
        self,
        chunk_id: str,
        status: str,
        processed_units: int,
        total_units: int,
        detail: str = "",
    ) -> None:
        """Update one chunk inside the current structured parallel-work snapshot."""

    def clear_parallel_work(self) -> None:
        """Clear any structured parallel-work snapshot for the current stage."""

    def is_canceled(self) -> bool:
        """Return whether the task has been canceled by the user."""


class PipelineStageRunner(Protocol):
    """Small stage contract that keeps stage implementations replaceable."""

    stage: PipelineStage

    def run(
        self,
        context: PipelineContext,
        config: PipelineConfig,
        reporter: ProgressReporter,
    ) -> ArtifactRecord:
        """Execute the stage and return artifact metadata for the produced output."""


def report_rows(
    reporter: ProgressReporter,
    total_rows: int,
    message_template: str,
) -> None:
    """Emit stable row-based progress updates for raster stages."""

    for row_index in range(total_rows):
        reporter.advance(
            1,
            message_template.format(index=row_index + 1, total=total_rows),
        )


def build_row_progress_callback(
    reporter: ProgressReporter,
    total_units: int,
) -> Callable[[str], None]:
    """Create a callback that advances stage progress once per emitted unit."""

    emitted_units = 0

    def _callback(message: str) -> None:
        nonlocal emitted_units
        if emitted_units >= total_units:
            return
        emitted_units += 1
        reporter.advance(1, message)

    return _callback


def build_heartbeat_callback(
    reporter: ProgressReporter,
    min_interval_seconds: float = 0.75,
) -> Callable[[str], None]:
    """Emit time-throttled liveness updates for long tight loops."""

    last_emitted_at = 0.0

    def _callback(message: str) -> None:
        nonlocal last_emitted_at
        now = monotonic()
        if now - last_emitted_at < min_interval_seconds:
            return

        last_emitted_at = now
        reporter.heartbeat(message)

    return _callback


class InputStageRunner:
    """Validate and preview the task input raster."""

    stage = PipelineStage.IO

    def run(
        self,
        context: PipelineContext,
        config: PipelineConfig,
        reporter: ProgressReporter,
    ) -> ArtifactRecord:
        stage_units = 5 if context.mask_path is not None else 4
        reporter.begin_stage(self.stage, stage_units, "Resolving input terrain path.")
        reporter.advance(1, f"Resolved input path: {context.input_path}")

        if not context.input_path.exists():
            raise FileNotFoundError(f"Input terrain file was not found: {context.input_path}")

        reporter.advance(1, "Input terrain file exists.")
        with Image.open(context.input_path) as image:
            width, height = image.size
            mode = image.mode

        pipeline_logger.info(
            "Opened terrain input '%s' with size=%sx%s and mode=%s.",
            context.input_path,
            width,
            height,
            mode,
        )
        reporter.advance(1, f"Detected raster size {width}x{height} with mode {mode}.")

        applied_user_mask: np.ndarray | None = None
        if context.mask_path is not None:
            user_mask = load_user_mask_for_shape(context, (height, width))
            reporter.publish_artifact(persist_user_mask_artifact(context, user_mask, self.stage))
            if config.preprocess.use_mask:
                applied_user_mask = user_mask
                reporter.advance(
                    1,
                    f"已加载用户遮罩 {context.mask_path.name}，并将从输入阶段开始参与后续计算。",
                )
            else:
                reporter.advance(
                    1,
                    f"检测到用户遮罩 {context.mask_path.name}，但当前任务未启用“用户遮罩”开关。",
                )

        preview_path = context.task_directory / "input_preview.png"
        with Image.open(context.input_path) as image:
            preview_image = self._to_preview_image(image, applied_user_mask)
            save_preview_image(preview_image, preview_path, int(config.preview_max_side))
        reporter.advance(1, f"Saved input preview to {preview_path}.")
        reporter.complete_stage("Input inspection completed.")

        artifact = ArtifactRecord(
            key="input_preview",
            label=get_artifact_label("input_preview"),
            stage=self.stage,
            status=ArtifactStatus.READY,
            path=to_project_relative_path(context.input_path),
            preview_path=to_project_relative_path(preview_path),
            width=width,
            height=height,
        )
        reporter.publish_artifact(artifact)
        return artifact

    def _to_preview_image(
        self,
        image: Image.Image,
        user_mask: np.ndarray | None,
    ) -> Image.Image:
        """Normalize input imagery into one stable preview format and optionally apply the user mask."""

        if user_mask is None:
            if image.mode == "RGB":
                return image.copy()

            if image.mode == "L":
                return image.convert("RGB")

            return image.convert("RGB")

        preview_rgba = image.convert("RGBA")
        preview_array = np.asarray(preview_rgba, dtype=np.uint8).copy()
        preview_array[~user_mask] = np.asarray((0, 0, 0, 0), dtype=np.uint8)
        return Image.fromarray(preview_array, mode="RGBA")


class PreprocessStageRunner:
    """Convert the input raster into a normalized terrain surface."""

    stage = PipelineStage.PREPROCESS

    def run(
        self,
        context: PipelineContext,
        config: PipelineConfig,
        reporter: ProgressReporter,
    ) -> ArtifactRecord:
        preview_path = context.task_directory / "terrain_preprocessed.png"
        raw_output_path = context.task_directory / "terrain_preprocessed.npy"
        original_for_flow_path = context.task_directory / "terrain_original_for_flow.npy"
        fill_depth_path = context.task_directory / "fill_depth.npy"
        fill_depth_preview_path = context.task_directory / "fill_depth.png"
        valid_mask_output_path = context.task_directory / "valid_mask.npy"
        ensure_parent_directory(preview_path)

        step_started_at = monotonic()
        with Image.open(context.input_path) as image:
            height_array = load_height_array(image)
        self._log_step_timing(context, "load_height_array", step_started_at)
        stage_rows = int(height_array.shape[0])
        estimated_units = 10
        if config.preprocess.smooth and config.preprocess.smooth_kernel_size > 1:
            estimated_units += stage_rows
        if config.preprocess.use_auto_mask:
            estimated_units += stage_rows * 6 + 16
        if config.preprocess.fill_sinks:
            estimated_units += (stage_rows + 11) * DEFAULT_FILL_SINK_MAX_ITERATIONS
        estimated_units = max(12, estimated_units)
        reporter.begin_stage(self.stage, estimated_units, "Loading terrain raster for preprocessing.")
        reporter.advance(1, "Converted source image into a single-channel terrain raster.")

        valid_mask = build_valid_mask(
            height_array,
            config.preprocess.preserve_nodata,
            config.preprocess.nodata_value,
        )
        step_started_at = monotonic()
        height_array = apply_height_mapping(height_array, config.preprocess.height_mapping)
        self._log_step_timing(context, "height_mapping", step_started_at)
        reporter.advance(1, self._build_height_mapping_message(config))
        step_started_at = monotonic()
        valid_mask = self._apply_optional_user_mask(height_array, valid_mask, context, config, reporter)
        self._log_step_timing(context, "user_mask", step_started_at)
        step_started_at = monotonic()
        valid_mask = self._apply_optional_auto_mask(height_array, valid_mask, context, config, reporter)
        self._log_step_timing(context, "auto_mask", step_started_at)

        roi = self._build_processing_roi(valid_mask, config)
        roi_slices = (slice(roi[0], roi[1]), slice(roi[2], roi[3]))
        work_height_array = height_array[roi_slices].astype(np.float32, copy=True)
        work_valid_mask = valid_mask[roi_slices].astype(bool, copy=True)
        if roi != (0, height_array.shape[0], 0, height_array.shape[1]):
            reporter.advance(
                1,
                (
                    "ROI 优化："
                    f"仅处理有效区域行 {roi[0] + 1}-{roi[1]}、列 {roi[2] + 1}-{roi[3]}，"
                    f"原图 {height_array.shape[1]}x{height_array.shape[0]}。"
                ),
            )

        step_started_at = monotonic()
        work_height_array = self._apply_optional_smoothing(work_height_array, work_valid_mask, config, reporter)
        self._log_step_timing(context, "smoothing", step_started_at)
        original_for_flow_array = height_array.astype(np.float32, copy=True)
        original_for_flow_array[roi_slices] = work_height_array
        np.save(original_for_flow_path, original_for_flow_array.astype(np.float32))
        step_started_at = monotonic()
        work_height_array, work_fill_depth = self._apply_optional_sink_fill(
            work_height_array,
            work_valid_mask,
            context,
            config,
            reporter,
        )
        self._log_step_timing(context, "sink_fill", step_started_at)
        height_array = height_array.astype(np.float32, copy=True)
        height_array[roi_slices] = work_height_array
        fill_depth = np.zeros_like(height_array, dtype=np.float32)
        if work_fill_depth is not None:
            fill_depth[roi_slices] = work_fill_depth

        preview_image = terrain_preview_image(height_array, valid_mask)
        save_preview_image(preview_image, preview_path, int(config.preview_max_side))
        np.save(raw_output_path, height_array.astype(np.float32))
        np.save(valid_mask_output_path, valid_mask.astype(np.uint8))
        if fill_depth is not None:
            np.save(fill_depth_path, fill_depth.astype(np.float32))
            save_preview_image(
                fill_depth_preview_image(fill_depth, valid_mask),
                fill_depth_preview_path,
                int(config.preview_max_side),
            )
            reporter.publish_artifact(
                ArtifactRecord(
                    key="fill_depth",
                    label=get_artifact_label("fill_depth"),
                    stage=self.stage,
                    status=ArtifactStatus.READY,
                    path=to_project_relative_path(fill_depth_path),
                    preview_path=to_project_relative_path(fill_depth_preview_path),
                    width=int(fill_depth.shape[1]),
                    height=int(fill_depth.shape[0]),
                )
            )
        reporter.advance(1, terrain_statistics_message(height_array, valid_mask))
        reporter.advance(1, f"Saved preprocessing artifacts to {preview_path}.")
        reporter.complete_stage("Terrain preprocessing completed.")
        pipeline_logger.info("Preprocessed terrain written to '%s'.", preview_path)

        artifact = ArtifactRecord(
            key="terrain_preprocessed",
            label=get_artifact_label("terrain_preprocessed"),
            stage=self.stage,
            status=ArtifactStatus.READY,
            path=to_project_relative_path(raw_output_path),
            preview_path=to_project_relative_path(preview_path),
            width=int(height_array.shape[1]),
            height=int(height_array.shape[0]),
        )
        reporter.publish_artifact(artifact)
        return artifact

    def _log_step_timing(
        self,
        context: PipelineContext,
        step_name: str,
        started_at: float,
    ) -> None:
        """Write one preprocess timing line for large-raster diagnosis."""

        elapsed_seconds = monotonic() - started_at
        pipeline_logger.info(
            "task_id=%s | stage=preprocess | step=%s | elapsed=%.3fs",
            context.task_id,
            step_name,
            elapsed_seconds,
        )

    def _build_processing_roi(
        self,
        valid_mask: np.ndarray,
        config: PipelineConfig,
    ) -> tuple[int, int, int, int]:
        """Return a padded valid-mask bounding box for expensive local preprocessing."""

        height, width = valid_mask.shape
        valid_rows, valid_columns = np.nonzero(valid_mask)
        if valid_rows.size == 0 or valid_columns.size == 0:
            return 0, height, 0, width

        smooth_radius = (
            int(config.preprocess.smooth_kernel_size) // 2
            if config.preprocess.smooth and config.preprocess.smooth_kernel_size > 1
            else 0
        )
        margin = max(2, smooth_radius + 2)
        row_start = max(0, int(valid_rows.min()) - margin)
        row_end = min(height, int(valid_rows.max()) + margin + 1)
        column_start = max(0, int(valid_columns.min()) - margin)
        column_end = min(width, int(valid_columns.max()) + margin + 1)
        return row_start, row_end, column_start, column_end

    def _apply_optional_auto_mask(
        self,
        height_array: np.ndarray,
        valid_mask: np.ndarray,
        context: PipelineContext,
        config: PipelineConfig,
        reporter: ProgressReporter,
    ) -> np.ndarray:
        """Generate and publish one automatic validity mask when the toggle is enabled."""

        if not config.preprocess.use_auto_mask:
            reporter.advance(1, "已跳过自动遮罩生成。")
            return valid_mask

        auto_mask = generate_auto_mask(
            height_array,
            valid_mask,
            enabled=config.preprocess.use_auto_mask,
            border_sensitivity=config.preprocess.auto_mask_border_sensitivity,
            texture_sensitivity=config.preprocess.auto_mask_texture_sensitivity,
            min_region_size=config.preprocess.auto_mask_min_region_size,
            progress_callback=build_row_progress_callback(reporter, int(height_array.shape[0]) * 6 + 16),
            heartbeat_callback=build_heartbeat_callback(reporter),
        )
        auto_mask_path = context.task_directory / "auto_mask.npy"
        auto_mask_preview_path = context.task_directory / "auto_mask.png"
        np.save(auto_mask_path, auto_mask.astype(np.uint8))
        save_preview_image(auto_mask_preview_image(auto_mask), auto_mask_preview_path, int(config.preview_max_side))
        merged_mask = valid_mask & auto_mask
        reporter.advance(
            1,
            (
                "自动遮罩生成完成："
                f"有效像素 {int(merged_mask.sum())}，"
                f"边界敏感度 {config.preprocess.auto_mask_border_sensitivity:.2f}，"
                f"纹理敏感度 {config.preprocess.auto_mask_texture_sensitivity:.2f}。"
            ),
        )

        artifact = ArtifactRecord(
            key="auto_mask",
            label=get_artifact_label("auto_mask"),
            stage=self.stage,
            status=ArtifactStatus.READY,
            path=to_project_relative_path(auto_mask_path),
            preview_path=to_project_relative_path(auto_mask_preview_path),
            width=int(auto_mask.shape[1]),
            height=int(auto_mask.shape[0]),
        )
        reporter.publish_artifact(artifact)
        return merged_mask

    def _build_height_mapping_message(self, config: PipelineConfig) -> str:
        """Describe the selected gray-to-height direction in one log line."""

        if config.preprocess.height_mapping == "bright_is_high":
            return "Applied height mapping: bright pixels are treated as higher terrain."

        return "Applied height mapping: dark pixels are treated as higher terrain."

    def _apply_optional_smoothing(
        self,
        height_array: np.ndarray,
        valid_mask: np.ndarray,
        config: PipelineConfig,
        reporter: ProgressReporter,
    ) -> np.ndarray:
        """Apply a small box blur while keeping invalid pixels untouched."""

        if not config.preprocess.smooth or config.preprocess.smooth_kernel_size <= 1:
            reporter.advance(1, "Skipped terrain smoothing.")
            return height_array

        smoothed_array = apply_box_smoothing(
            height_array,
            valid_mask,
            config.preprocess.smooth,
            config.preprocess.smooth_kernel_size,
            progress_callback=build_row_progress_callback(reporter, int(height_array.shape[0])),
            heartbeat_callback=build_heartbeat_callback(reporter),
            parallel_work_callback=reporter.set_parallel_work,
            parallel_chunk_callback=reporter.update_parallel_chunk,
        )
        reporter.advance(
            1,
            f"Applied box smoothing with kernel size {config.preprocess.smooth_kernel_size}.",
        )
        return smoothed_array

    def _apply_optional_sink_fill(
        self,
        height_array: np.ndarray,
        valid_mask: np.ndarray,
        context: PipelineContext,
        config: PipelineConfig,
        reporter: ProgressReporter,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Repair closed sinks and no-outlet flat basins during preprocessing."""

        if not config.preprocess.fill_sinks:
            reporter.advance(1, "Skipped closed-sink and no-outlet flat repair.")
            return height_array, np.zeros_like(height_array, dtype=np.float32)

        pixel_count = int(height_array.shape[0] * height_array.shape[1])
        algorithm = config.preprocess.fill_sink_algorithm
        use_priority_flood = algorithm == "priority_flood" or (
            algorithm == "auto"
            and rust_priority_flood_available()
            and pixel_count >= int(config.preprocess.fast_fill_min_pixels)
        )
        max_fill_depth = config.preprocess.max_fill_depth
        if config.preprocess.deep_basin_mode == "preserve" and max_fill_depth is None:
            max_fill_depth = 0.0

        if use_priority_flood:
            filled_array, fill_depth = fill_depressions_priority_flood(
                height_array,
                valid_mask,
                max_fill_depth=max_fill_depth,
                progress_callback=build_row_progress_callback(reporter, max(16, int(height_array.shape[0]))),
                heartbeat_callback=build_heartbeat_callback(reporter),
            )
            changed_cells = int((fill_depth > 0).sum())
            max_depth = float(fill_depth.max(initial=0.0))
            mean_depth = float(fill_depth[fill_depth > 0].mean()) if changed_cells > 0 else 0.0
            reporter.advance(
                1,
                (
                    "快速填洼完成："
                    f"algorithm=priority_flood，像素 {pixel_count}，"
                    f"修复 {changed_cells}，最大填深 {max_depth:.3f}，平均填深 {mean_depth:.3f}，"
                    f"deep_basin_mode={config.preprocess.deep_basin_mode}。"
                ),
            )
            pipeline_logger.info(
                (
                    "task_id=%s | stage=preprocess | step=sink_fill_audit | "
                    "algorithm=priority_flood | pixels=%s | changed=%s | max_depth=%.3f | mean_depth=%.3f"
                ),
                context.task_id,
                pixel_count,
                changed_cells,
                max_depth,
                mean_depth,
            )
            return filled_array, fill_depth

        filled_array, iterations_used = fill_local_sinks(
            height_array,
            valid_mask,
            config.preprocess.fill_sinks,
            progress_callback=build_row_progress_callback(
                reporter,
                (int(height_array.shape[0]) + 11) * DEFAULT_FILL_SINK_MAX_ITERATIONS,
            ),
            heartbeat_callback=build_heartbeat_callback(reporter),
            parallel_work_callback=reporter.set_parallel_work,
            parallel_chunk_callback=reporter.update_parallel_chunk,
        )
        reporter.advance(
            1,
            f"Repaired closed sinks and no-outlet flat basins in {iterations_used} iteration(s).",
        )
        fill_depth = np.where(
            valid_mask,
            np.maximum(filled_array.astype(np.float32, copy=False) - height_array.astype(np.float32, copy=False), 0.0),
            0.0,
        ).astype(np.float32, copy=False)
        pipeline_logger.info(
            (
                "task_id=%s | stage=preprocess | step=sink_fill_audit | "
                "algorithm=legacy | iterations=%s | changed=%s | max_depth=%.3f"
            ),
            context.task_id,
            iterations_used,
            int((fill_depth > 0).sum()),
            float(fill_depth.max(initial=0.0)),
        )
        return filled_array, fill_depth

    def _apply_optional_user_mask(
        self,
        height_array: np.ndarray,
        valid_mask: np.ndarray,
        context: PipelineContext,
        config: PipelineConfig,
        reporter: ProgressReporter,
    ) -> np.ndarray:
        """
        Merge an optional user-provided mask into the working valid-data mask.

        The mask feature is opt-in. If the user does not provide a mask, or the
        mask toggle is off, preprocessing behavior stays unchanged.
        """

        if not config.preprocess.use_mask or context.mask_path is None:
            reporter.advance(1, "未应用用户遮罩。")
            return valid_mask

        user_mask = load_user_mask_for_shape(context, height_array.shape)

        merged_mask = valid_mask & user_mask
        reporter.advance(
            1,
            f"已在预处理起点并入用户遮罩 {context.mask_path.name}；有效像素 {int(merged_mask.sum())}。",
        )
        return merged_mask


class FlowDirectionStageRunner:
    """Compute a conservative D8 flow-direction raster."""

    stage = PipelineStage.FLOW_DIRECTION

    def run(
        self,
        context: PipelineContext,
        config: PipelineConfig,
        reporter: ProgressReporter,
    ) -> ArtifactRecord:
        terrain_path = context.task_directory / "terrain_preprocessed.npy"
        texture_terrain_path = context.task_directory / "terrain_original_for_flow.npy"
        valid_mask_path = context.task_directory / "valid_mask.npy"
        preview_path = context.task_directory / "flow_direction.png"
        raw_output_path = context.task_directory / "flow_direction.npy"

        if not terrain_path.exists():
            raise FileNotFoundError(f"Preprocessed terrain raster was not found: {terrain_path}")
        if not valid_mask_path.exists():
            raise FileNotFoundError(f"Preprocessed valid mask was not found: {valid_mask_path}")

        terrain = np.load(terrain_path).astype(np.float32)
        texture_terrain = (
            np.load(texture_terrain_path).astype(np.float32)
            if texture_terrain_path.exists()
            else terrain
        )
        valid_mask = np.load(valid_mask_path).astype(bool)
        total_rows = int(terrain.shape[0])
        total_nodes = int(terrain.shape[0] * terrain.shape[1])
        residual_units = max(8, min(64, total_nodes))
        cycle_units = max(4, min(32, total_nodes))
        reporter.begin_stage(
            self.stage,
            total_rows * 2 + residual_units + cycle_units,
            "Computing D8 flow directions: 1/4 strict downhill, 2/4 flat routing, 3/4 fallback repair, 4/4 cycle cleanup.",
        )
        direction_array = compute_d8_flow_directions(
            terrain,
            valid_mask=valid_mask,
            texture_height_array=texture_terrain,
            use_rust_kernel=config.flow_direction.use_rust_kernel,
            slope_weight=config.flow_direction.slope_weight,
            flat_escape_weight=config.flow_direction.flat_escape_weight,
            outlet_proximity_weight=config.flow_direction.outlet_proximity_weight,
            continuity_weight=config.flow_direction.continuity_weight,
            flat_outlet_length_weight=config.flow_direction.flat_outlet_length_weight,
            flat_outlet_distance_weight=config.flow_direction.flat_outlet_distance_weight,
            strict_progress_callback=build_row_progress_callback(reporter, total_rows),
            flat_progress_callback=build_row_progress_callback(reporter, total_rows),
            residual_progress_callback=build_row_progress_callback(reporter, residual_units),
            strict_heartbeat_callback=build_heartbeat_callback(reporter),
            flat_heartbeat_callback=build_heartbeat_callback(reporter),
            residual_heartbeat_callback=build_heartbeat_callback(reporter),
            cycle_progress_callback=build_row_progress_callback(reporter, cycle_units),
            cycle_heartbeat_callback=build_heartbeat_callback(reporter),
            parallel_work_callback=reporter.set_parallel_work,
            parallel_chunk_callback=reporter.update_parallel_chunk,
            clear_parallel_work_callback=reporter.clear_parallel_work,
        )

        preview_image = direction_preview_image(direction_array)
        save_preview_image(preview_image, preview_path, int(config.preview_max_side))
        np.save(raw_output_path, direction_array.astype(np.int8))
        reporter.complete_stage("D8 flow-direction stage completed.")
        pipeline_logger.info("Flow direction raster written to '%s'.", raw_output_path)

        artifact = ArtifactRecord(
            key="flow_direction",
            label=get_artifact_label("flow_direction"),
            stage=self.stage,
            status=ArtifactStatus.READY,
            path=to_project_relative_path(raw_output_path),
            preview_path=to_project_relative_path(preview_path),
            width=int(direction_array.shape[1]),
            height=int(direction_array.shape[0]),
        )
        reporter.publish_artifact(artifact)
        return artifact


class FlowAccumulationStageRunner:
    """Compute upstream flow accumulation from the D8 direction raster."""

    stage = PipelineStage.FLOW_ACCUMULATION

    def run(
        self,
        context: PipelineContext,
        config: PipelineConfig,
        reporter: ProgressReporter,
    ) -> ArtifactRecord:
        direction_path = context.task_directory / "flow_direction.npy"
        preview_path = context.task_directory / "flow_accumulation.png"
        raw_output_path = context.task_directory / "flow_accumulation.npy"

        if not direction_path.exists():
            raise FileNotFoundError(f"Flow direction raster was not found: {direction_path}")

        direction_array = np.load(direction_path).astype(np.int16)
        total_rows = int(direction_array.shape[0])
        total_nodes = int(direction_array.shape[0] * direction_array.shape[1])
        propagation_units = max(16, min(128, total_nodes))
        reporter.begin_stage(
            self.stage,
            total_rows + propagation_units,
            "Building flow-dependency graph and propagating accumulation.",
        )
        accumulation = compute_flow_accumulation(
            direction_array,
            index_progress_callback=build_row_progress_callback(reporter, total_rows),
            propagate_progress_callback=build_row_progress_callback(reporter, propagation_units),
            heartbeat_callback=build_heartbeat_callback(reporter),
            use_rust_kernel=config.flow_accumulation.use_rust_kernel,
        )

        preview_image = accumulation_preview_image(accumulation, config.flow_accumulation.normalize)
        save_preview_image(preview_image, preview_path, int(config.preview_max_side))
        np.save(raw_output_path, accumulation.astype(np.float32))
        reporter.complete_stage("Flow accumulation stage completed.")
        pipeline_logger.info("Flow accumulation raster written to '%s'.", raw_output_path)

        artifact = ArtifactRecord(
            key="flow_accumulation",
            label=get_artifact_label("flow_accumulation"),
            stage=self.stage,
            status=ArtifactStatus.READY,
            path=to_project_relative_path(raw_output_path),
            preview_path=to_project_relative_path(preview_path),
            width=int(accumulation.shape[1]),
            height=int(accumulation.shape[0]),
        )
        reporter.publish_artifact(artifact)
        return artifact


class ChannelExtractStageRunner:
    """Threshold the accumulation raster into a binary channel mask."""

    stage = PipelineStage.CHANNEL_EXTRACT

    def run(
        self,
        context: PipelineContext,
        config: PipelineConfig,
        reporter: ProgressReporter,
    ) -> ArtifactRecord:
        accumulation_path = context.task_directory / "flow_accumulation.npy"
        valid_mask_path = context.task_directory / "valid_mask.npy"
        direction_path = context.task_directory / "flow_direction.npy"
        preview_path = context.task_directory / "channel_mask.png"
        raw_output_path = context.task_directory / "channel_mask.npy"

        if not accumulation_path.exists():
            raise FileNotFoundError(f"Flow accumulation raster was not found: {accumulation_path}")
        if not valid_mask_path.exists():
            raise FileNotFoundError(f"Preprocessed valid mask was not found: {valid_mask_path}")
        if not direction_path.exists():
            raise FileNotFoundError(f"Flow direction raster was not found: {direction_path}")

        accumulation = np.load(accumulation_path).astype(np.float32)
        valid_mask = np.load(valid_mask_path).astype(bool)
        direction_array = np.load(direction_path).astype(np.int16)
        threshold = float(config.channel_extract.accumulation_threshold)
        channel_length_threshold = int(config.channel_extract.channel_length_threshold)
        stage_units = int(accumulation.shape[0]) * 4 + 32
        reporter.begin_stage(
            self.stage,
            stage_units,
            (
                "Extracting channels with accumulation threshold "
                f"{threshold:.2f}, trimming edge-following segments, and pruning channels shorter than "
                f"{channel_length_threshold} pixels."
            ),
        )
        channel_mask = build_channel_mask(
            accumulation,
            threshold,
            valid_mask=valid_mask,
            direction_array=direction_array,
            channel_length_threshold=channel_length_threshold,
            progress_callback=build_row_progress_callback(reporter, stage_units),
            heartbeat_callback=build_heartbeat_callback(reporter),
            parallel_work_callback=reporter.set_parallel_work,
            parallel_chunk_callback=reporter.update_parallel_chunk,
        )

        preview_image = channel_preview_image(channel_mask)
        saved_preview_image = save_preview_image(preview_image, preview_path, int(config.preview_max_side))
        np.save(raw_output_path, channel_mask.astype(np.uint8))
        np.save(context.output_path.with_suffix(".npy"), channel_mask.astype(np.uint8))
        final_output_image = (
            preview_image
            if context.output_path.suffix.lower() in {".png", ".webp"}
            else preview_image.convert("RGB")
        )
        final_output_image.save(context.output_path)
        reporter.complete_stage("Channel extraction stage completed.")
        pipeline_logger.info("Channel mask written to '%s'.", raw_output_path)

        artifact = ArtifactRecord(
            key="channel_mask",
            label=get_artifact_label("channel_mask"),
            stage=self.stage,
            status=ArtifactStatus.READY,
            path=to_project_relative_path(raw_output_path),
            preview_path=to_project_relative_path(preview_path),
            width=int(channel_mask.shape[1]),
            height=int(channel_mask.shape[0]),
        )
        reporter.publish_artifact(artifact)
        return artifact


class ArtifactCapturingReporter:
    """Forward progress calls while remembering artifacts emitted inside a stage."""

    def __init__(
        self,
        delegate: ProgressReporter,
        captured_artifacts: list[ArtifactRecord],
    ) -> None:
        self._delegate = delegate
        self._captured_artifacts = captured_artifacts

    def begin_stage(self, stage: PipelineStage, total_units: int, message: str) -> None:
        self._delegate.begin_stage(stage, total_units, message)

    def advance(self, units: int = 1, message: str = "") -> None:
        self._delegate.advance(units, message)

    def complete_stage(self, message: str = "") -> None:
        self._delegate.complete_stage(message)

    def log(self, message: str) -> None:
        self._delegate.log(message)

    def publish_artifact(self, artifact: ArtifactRecord) -> None:
        self._captured_artifacts.append(artifact)
        self._delegate.publish_artifact(artifact)

    def heartbeat(self, message: str = "", force: bool = False) -> None:
        self._delegate.heartbeat(message, force=force)

    def set_parallel_work(
        self,
        label: str,
        strategy: str,
        chunks: list[tuple[str, str, int]],
    ) -> None:
        self._delegate.set_parallel_work(label, strategy, chunks)

    def update_parallel_chunk(
        self,
        chunk_id: str,
        status: str,
        processed_units: int,
        total_units: int,
        detail: str = "",
    ) -> None:
        self._delegate.update_parallel_chunk(
            chunk_id,
            status,
            processed_units,
            total_units,
            detail,
        )

    def clear_parallel_work(self) -> None:
        self._delegate.clear_parallel_work()

    def is_canceled(self) -> bool:
        return self._delegate.is_canceled()


class RiverPipeline:
    """Top-level staged pipeline entry point."""

    def __init__(self) -> None:
        self._stages: tuple[PipelineStageRunner, ...] = (
            InputStageRunner(),
            PreprocessStageRunner(),
            FlowDirectionStageRunner(),
            FlowAccumulationStageRunner(),
            ChannelExtractStageRunner(),
        )

    def run(
        self,
        task_id: str,
        request: RiverTaskRequest,
        reporter: ProgressReporter,
        existing_result: PipelineResult | None = None,
    ) -> PipelineResult:
        """Execute the pipeline and return the explicit artifact registry."""

        context = build_pipeline_context(task_id, request)
        config = request.config
        result = (
            existing_result.model_copy(deep=True)
            if existing_result is not None
            else build_default_pipeline_result(context)
        )
        write_pipeline_metadata(result, context.metadata_path)
        start_stage = request.start_stage or PIPELINE_STAGE_SEQUENCE[0]
        end_stage = request.end_stage or PIPELINE_STAGE_SEQUENCE[-1]
        start_index = get_stage_index(start_stage)
        end_index = get_stage_index(end_stage)
        pipeline_logger.info(
            (
                "Pipeline run started for task_id='%s' input='%s' output='%s' "
                "total_tiles=%s start_stage=%s end_stage=%s."
            ),
            task_id,
            context.input_path,
            context.output_path,
            config.total_tiles,
            start_stage.value,
            end_stage.value,
        )
        for stage_index, stage_runner in enumerate(self._stages):
            if reporter.is_canceled():
                pipeline_logger.warning("Pipeline run interrupted because the task was canceled.")
                break

            if stage_index < start_index:
                reporter.log(f"继承已有阶段产物，跳过 {stage_runner.stage.value}。")
                continue
            if stage_index > end_index:
                break

            captured_artifacts: list[ArtifactRecord] = []
            stage_reporter = ArtifactCapturingReporter(reporter, captured_artifacts)
            artifact = stage_runner.run(context, config, stage_reporter)
            for captured_artifact in captured_artifacts:
                result = update_result_with_artifact(
                    result,
                    captured_artifact,
                    context.metadata_path,
                )
            result = update_result_with_artifact(result, artifact, context.metadata_path)
            if stage_index == end_index:
                reporter.log(f"任务已按请求在 {stage_runner.stage.value} 阶段结束。")
                break

        pipeline_logger.info("Pipeline run completed with artifacts: %s", result.artifacts)
        return result
