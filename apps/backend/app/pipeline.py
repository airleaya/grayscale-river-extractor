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
    PipelineConfig,
    PipelineResult,
    PipelineStage,
    RiverTaskRequest,
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
    fill_local_sinks,
    generate_auto_mask,
    load_height_array,
    load_mask_array,
    terrain_preview_image,
    terrain_statistics_message,
    DEFAULT_FILL_SINK_MAX_ITERATIONS,
)


pipeline_logger = get_logger("river.pipeline")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_LABELS = {
    "input_preview": "输入图像",
    "auto_mask": "自动遮罩",
    "terrain_preprocessed": "预处理地形",
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
        del config
        reporter.begin_stage(self.stage, 3, "Resolving input terrain path.")
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
        reporter.complete_stage("Input inspection completed.")

        preview_path = context.task_directory / "input_preview.png"
        with Image.open(context.input_path) as image:
            preview_image = self._to_preview_image(image)
            preview_image.save(preview_path)

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

    def _to_preview_image(self, image: Image.Image) -> Image.Image:
        """Normalize input imagery into a stable RGB preview format."""

        if image.mode == "RGB":
            return image.copy()

        if image.mode == "L":
            return image.convert("RGB")

        return image.convert("RGB")


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
        valid_mask_output_path = context.task_directory / "valid_mask.npy"
        ensure_parent_directory(preview_path)

        with Image.open(context.input_path) as image:
            height_array = load_height_array(image)
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
        height_array = apply_height_mapping(height_array, config.preprocess.height_mapping)
        reporter.advance(1, self._build_height_mapping_message(config))
        valid_mask = self._apply_optional_auto_mask(height_array, valid_mask, context, config, reporter)
        valid_mask = self._apply_optional_user_mask(height_array, valid_mask, context, config, reporter)

        height_array = self._apply_optional_smoothing(height_array, valid_mask, config, reporter)
        height_array = self._apply_optional_sink_fill(height_array, valid_mask, config, reporter)

        preview_image = terrain_preview_image(height_array, valid_mask)
        preview_image.save(preview_path)
        np.save(raw_output_path, height_array.astype(np.float32))
        np.save(valid_mask_output_path, valid_mask.astype(np.uint8))
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
        auto_mask_preview_image(auto_mask).save(auto_mask_preview_path)
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
        config: PipelineConfig,
        reporter: ProgressReporter,
    ) -> np.ndarray:
        """Repair closed sinks and no-outlet flat basins during preprocessing."""

        if not config.preprocess.fill_sinks:
            reporter.advance(1, "Skipped closed-sink and no-outlet flat repair.")
            return height_array

        filled_array, iterations_used = fill_local_sinks(
            height_array,
            valid_mask,
            config.preprocess.fill_sinks,
            progress_callback=build_row_progress_callback(
                reporter,
                (int(height_array.shape[0]) + 11) * DEFAULT_FILL_SINK_MAX_ITERATIONS,
            ),
            heartbeat_callback=build_heartbeat_callback(reporter),
        )
        reporter.advance(
            1,
            f"Repaired closed sinks and no-outlet flat basins in {iterations_used} iteration(s).",
        )
        return filled_array

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

        if not context.mask_path.exists():
            raise FileNotFoundError(f"用户遮罩栅格未找到：{context.mask_path}")

        with Image.open(context.mask_path) as image:
            user_mask = load_mask_array(image)

        if user_mask.shape != height_array.shape:
            raise ValueError(
                "用户遮罩栅格尺寸与地形栅格不一致："
                f"{user_mask.shape} != {height_array.shape}"
            )

        merged_mask = valid_mask & user_mask
        reporter.advance(
            1,
            f"已应用用户遮罩 {context.mask_path.name}；有效像素 {int(merged_mask.sum())}。",
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
        valid_mask_path = context.task_directory / "valid_mask.npy"
        preview_path = context.task_directory / "flow_direction.png"
        raw_output_path = context.task_directory / "flow_direction.npy"

        if not terrain_path.exists():
            raise FileNotFoundError(f"Preprocessed terrain raster was not found: {terrain_path}")
        if not valid_mask_path.exists():
            raise FileNotFoundError(f"Preprocessed valid mask was not found: {valid_mask_path}")

        terrain = np.load(terrain_path).astype(np.float32)
        valid_mask = np.load(valid_mask_path).astype(bool)
        total_rows = int(terrain.shape[0])
        total_nodes = int(terrain.shape[0] * terrain.shape[1])
        residual_units = max(8, min(64, total_nodes))
        reporter.begin_stage(
            self.stage,
            total_rows * 2 + residual_units,
            "Computing D8 flow directions: 1/3 strict downhill, 2/3 flat routing, 3/3 fallback repair.",
        )
        direction_array = compute_d8_flow_directions(
            terrain,
            valid_mask=valid_mask,
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
        )

        preview_image = direction_preview_image(direction_array)
        preview_image.save(preview_path)
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
        )

        preview_image = accumulation_preview_image(accumulation, config.flow_accumulation.normalize)
        preview_image.save(preview_path)
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
        preview_path = context.task_directory / "channel_mask.png"
        raw_output_path = context.task_directory / "channel_mask.npy"

        if not accumulation_path.exists():
            raise FileNotFoundError(f"Flow accumulation raster was not found: {accumulation_path}")
        if not valid_mask_path.exists():
            raise FileNotFoundError(f"Preprocessed valid mask was not found: {valid_mask_path}")

        accumulation = np.load(accumulation_path).astype(np.float32)
        valid_mask = np.load(valid_mask_path).astype(bool)
        threshold = float(config.channel_extract.accumulation_threshold)
        reporter.begin_stage(
            self.stage,
            int(accumulation.shape[0]),
            f"Extracting channels with accumulation threshold {threshold:.2f}.",
        )
        channel_mask = build_channel_mask(
            accumulation,
            threshold,
            valid_mask=valid_mask,
            progress_callback=build_row_progress_callback(reporter, int(accumulation.shape[0])),
            heartbeat_callback=build_heartbeat_callback(reporter),
        )

        preview_image = channel_preview_image(channel_mask)
        preview_image.save(preview_path)
        np.save(raw_output_path, channel_mask.astype(np.uint8))
        np.save(context.output_path.with_suffix(".npy"), channel_mask.astype(np.uint8))
        preview_image.save(context.output_path)
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

    def run(self, task_id: str, request: RiverTaskRequest, reporter: ProgressReporter) -> PipelineResult:
        """Execute the pipeline and return the explicit artifact registry."""

        context = build_pipeline_context(task_id, request)
        config = request.config
        result = build_default_pipeline_result(context)
        write_pipeline_metadata(result, context.metadata_path)
        pipeline_logger.info(
            "Pipeline run started for task_id='%s' input='%s' output='%s' total_tiles=%s.",
            task_id,
            context.input_path,
            context.output_path,
            config.total_tiles,
        )
        for stage_runner in self._stages:
            if reporter.is_canceled():
                pipeline_logger.warning("Pipeline run interrupted because the task was canceled.")
                break

            artifact = stage_runner.run(context, config, reporter)
            result = update_result_with_artifact(result, artifact, context.metadata_path)

        pipeline_logger.info("Pipeline run completed with artifacts: %s", result.artifacts)
        return result
