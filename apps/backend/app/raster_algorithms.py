"""
Pure raster algorithms used by the staged river pipeline.

This module deliberately keeps image math and array transforms separate from
stage orchestration so future optimization work can focus on one place.
"""

from __future__ import annotations

from collections import deque
from math import ceil
from time import monotonic
from typing import Callable

import numpy as np
from PIL import Image

from .rust_bridge import (
    compute_flat_outlet_drop_map_rust,
    compute_strict_d8_rust,
    label_connected_components_rust,
    label_equal_height_regions_rust,
    rust_kernel_available,
)

D8_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)
D8_DISTANCES = np.asarray(
    (1.0, np.sqrt(2.0), 1.0, np.sqrt(2.0), 1.0, np.sqrt(2.0), 1.0, np.sqrt(2.0)),
    dtype=np.float32,
)
DIRECTION_PREVIEW_PALETTE = np.asarray(
    (
        (54, 116, 181),
        (102, 170, 255),
        (70, 172, 124),
        (145, 206, 108),
        (214, 165, 61),
        (220, 116, 69),
        (188, 81, 83),
        (130, 98, 173),
    ),
    dtype=np.uint8,
)
DEFAULT_SLOPE_WEIGHT = 1.0
DEFAULT_FLAT_ESCAPE_WEIGHT = 0.6
DEFAULT_OUTLET_PROXIMITY_WEIGHT = 0.4
DEFAULT_CONTINUITY_WEIGHT = 0.3
DEFAULT_FLAT_OUTLET_LENGTH_WEIGHT = 0.35
DEFAULT_FLAT_OUTLET_DISTANCE_WEIGHT = 1.5
ProgressCallback = Callable[[str], None]
DEFAULT_FILL_SINK_MAX_ITERATIONS = 32
AUTO_MASK_MAX_ANALYSIS_SIDE = 1536
AUTO_MASK_MIN_DOWNSAMPLE_SCALE = 2
AUTO_MASK_COMPONENT_HEARTBEAT_INTERVAL = 4096


def _emit_throttled_heartbeat(
    heartbeat_callback: ProgressCallback | None,
    next_heartbeat_at: float,
    message: str,
    min_interval_seconds: float = 0.75,
) -> float:
    """Emit a heartbeat message at most once per interval during tight Python loops."""

    if heartbeat_callback is None:
        return next_heartbeat_at

    now = monotonic()
    if now < next_heartbeat_at:
        return next_heartbeat_at

    heartbeat_callback(message)
    return now + min_interval_seconds


def load_height_array(image: Image.Image) -> np.ndarray:
    """Convert a source image into a float32 single-channel terrain raster."""

    grayscale = image.copy() if image.mode == "L" else image.convert("L")
    return np.asarray(grayscale, dtype=np.float32)


def load_mask_array(image: Image.Image) -> np.ndarray:
    """
    Convert a mask image into a boolean raster.

    Any non-zero pixel is treated as a valid processing area. This keeps the
    first version permissive and easy to prepare with common paint tools.
    """

    grayscale = image.copy() if image.mode == "L" else image.convert("L")
    return np.asarray(grayscale, dtype=np.uint8) > 0


def auto_mask_preview_image(mask_array: np.ndarray) -> Image.Image:
    """Convert one boolean mask into a stable grayscale preview image."""

    preview_array = np.where(mask_array, 255, 0).astype(np.uint8)
    return Image.fromarray(preview_array, mode="L")


def apply_height_mapping(
    height_array: np.ndarray,
    height_mapping: str,
) -> np.ndarray:
    """Normalize height direction so larger values always mean higher terrain."""

    if height_mapping == "bright_is_high":
        return height_array.copy()

    return 255.0 - height_array


def build_valid_mask(
    height_array: np.ndarray,
    preserve_nodata: bool,
    nodata_value: float | None,
) -> np.ndarray:
    """Build the valid-data mask used by later preprocessing helpers."""

    if not preserve_nodata or nodata_value is None:
        return np.ones_like(height_array, dtype=bool)

    return ~np.isclose(height_array, float(nodata_value))


def _collect_border_values(height_array: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Collect valid raster values from the outer border to model likely background."""

    if height_array.size == 0:
        return np.asarray([], dtype=np.float32)

    top = height_array[0, :][valid_mask[0, :]]
    bottom = height_array[-1, :][valid_mask[-1, :]]
    left = height_array[:, 0][valid_mask[:, 0]]
    right = height_array[:, -1][valid_mask[:, -1]]
    values = np.concatenate((top, bottom, left, right)).astype(np.float32, copy=False)
    return values if values.size > 0 else height_array[valid_mask].astype(np.float32, copy=False)


def _emit_row_progress(
    progress_callback: ProgressCallback | None,
    total_rows: int,
    message_template: str,
) -> None:
    """Replay row-level progress for vectorized stages."""

    if progress_callback is None:
        return

    for row_index in range(total_rows):
        progress_callback(message_template.format(index=row_index + 1, total=total_rows))


def _choose_auto_mask_downsample_scale(height: int, width: int) -> int:
    """Choose a downsample scale that caps auto-mask analysis cost on huge rasters."""

    max_side = max(height, width)
    if max_side <= AUTO_MASK_MAX_ANALYSIS_SIDE:
        return 1

    return max(
        AUTO_MASK_MIN_DOWNSAMPLE_SCALE,
        int(ceil(max_side / AUTO_MASK_MAX_ANALYSIS_SIDE)),
    )


def _downsample_auto_mask_inputs(
    height_array: np.ndarray,
    valid_mask: np.ndarray,
    scale: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample terrain and validity with valid-aware block aggregation."""

    if scale <= 1:
        return height_array.astype(np.float32, copy=True), valid_mask.astype(bool, copy=True)

    source_height, source_width = height_array.shape
    padded_height = int(ceil(source_height / scale) * scale)
    padded_width = int(ceil(source_width / scale) * scale)
    pad_rows = padded_height - source_height
    pad_columns = padded_width - source_width

    padded_values = np.pad(
        np.where(valid_mask, height_array, 0.0).astype(np.float32, copy=False),
        ((0, pad_rows), (0, pad_columns)),
        mode="constant",
        constant_values=0.0,
    )
    padded_valid = np.pad(
        valid_mask.astype(bool, copy=False),
        ((0, pad_rows), (0, pad_columns)),
        mode="constant",
        constant_values=False,
    )
    reshaped_values = padded_values.reshape(padded_height // scale, scale, padded_width // scale, scale)
    reshaped_valid = padded_valid.reshape(padded_height // scale, scale, padded_width // scale, scale)
    valid_counts = reshaped_valid.sum(axis=(1, 3)).astype(np.int32)
    value_sums = reshaped_values.sum(axis=(1, 3), dtype=np.float32)
    coarse_valid = valid_counts > 0
    coarse_height = np.divide(
        value_sums,
        np.maximum(valid_counts, 1),
        out=np.zeros_like(value_sums, dtype=np.float32),
        where=coarse_valid,
    ).astype(np.float32, copy=False)
    return coarse_height, coarse_valid


def _upsample_mask_to_shape(mask_array: np.ndarray, shape: tuple[int, int], scale: int) -> np.ndarray:
    """Upsample a coarse boolean mask back to the requested shape."""

    if scale <= 1:
        return mask_array.astype(bool, copy=True)

    expanded = np.repeat(np.repeat(mask_array.astype(bool, copy=False), scale, axis=0), scale, axis=1)
    target_height, target_width = shape
    return expanded[:target_height, :target_width]


def _scale_component_threshold(pixel_threshold: int, scale: int) -> int:
    """Convert a full-resolution area threshold into coarse-grid pixels."""

    if scale <= 1:
        return max(1, int(pixel_threshold))

    return max(1, int(ceil(float(pixel_threshold) / float(scale * scale))))


def _compute_gradient_scores(
    height_array: np.ndarray,
    valid_mask: np.ndarray,
    progress_callback: ProgressCallback | None = None,
    heartbeat_callback: ProgressCallback | None = None,
) -> np.ndarray:
    """Estimate low-gradient likelihood with vectorized D8 scans."""

    height, width = height_array.shape
    gradient = np.zeros_like(height_array, dtype=np.float32)
    padded_height = np.pad(height_array.astype(np.float32, copy=False), 1, mode="edge")
    padded_valid = np.pad(valid_mask.astype(bool, copy=False), 1, mode="constant", constant_values=False)
    center_height = padded_height[1 : height + 1, 1 : width + 1]
    center_valid = padded_valid[1 : height + 1, 1 : width + 1]
    next_heartbeat_at = monotonic() + 0.75

    for direction_index, (row_delta, column_delta) in enumerate(D8_OFFSETS):
        neighbor_height = padded_height[
            1 + row_delta : 1 + row_delta + height,
            1 + column_delta : 1 + column_delta + width,
        ]
        neighbor_valid = padded_valid[
            1 + row_delta : 1 + row_delta + height,
            1 + column_delta : 1 + column_delta + width,
        ]
        direction_delta = np.where(
            center_valid & neighbor_valid,
            np.abs(center_height - neighbor_height),
            0.0,
        ).astype(np.float32, copy=False)
        gradient = np.maximum(gradient, direction_delta)
        next_heartbeat_at = _emit_throttled_heartbeat(
            heartbeat_callback,
            next_heartbeat_at,
            f"自动遮罩梯度分析：正在处理第 {direction_index + 1}/8 个邻域方向。",
        )

    _emit_row_progress(
        progress_callback,
        height,
        "自动遮罩梯度分析：已完成 {index}/{total} 行。",
    )

    valid_gradients = gradient[valid_mask]
    if valid_gradients.size == 0:
        return np.zeros_like(height_array, dtype=np.float32)

    scale = float(np.percentile(valid_gradients, 95))
    if scale <= 1e-6:
        scale = 1.0

    return 1.0 - np.clip(gradient / scale, 0.0, 1.0)


def _compute_local_variance_scores(
    height_array: np.ndarray,
    valid_mask: np.ndarray,
    progress_callback: ProgressCallback | None = None,
    heartbeat_callback: ProgressCallback | None = None,
) -> np.ndarray:
    """Estimate low-variance likelihood with a 3x3 valid-aware window."""

    height, width = height_array.shape
    variance = np.zeros_like(height_array, dtype=np.float32)
    padded_height = np.pad(height_array, 1, mode="edge").astype(np.float32)
    padded_valid = np.pad(valid_mask.astype(np.float32), 1, mode="constant", constant_values=0.0)
    value_integral = np.pad(
        np.cumsum(np.cumsum(padded_height * padded_valid, axis=0), axis=1),
        ((1, 0), (1, 0)),
        mode="constant",
    )
    square_integral = np.pad(
        np.cumsum(np.cumsum((padded_height ** 2) * padded_valid, axis=0), axis=1),
        ((1, 0), (1, 0)),
        mode="constant",
    )
    valid_integral = np.pad(
        np.cumsum(np.cumsum(padded_valid, axis=0), axis=1),
        ((1, 0), (1, 0)),
        mode="constant",
    )
    next_heartbeat_at = monotonic() + 0.75

    for row_index in range(height):
        top = row_index
        bottom = row_index + 3
        left = np.arange(width, dtype=np.int32)
        right = left + 3
        window_sum = (
            value_integral[bottom, right]
            - value_integral[top, right]
            - value_integral[bottom, left]
            + value_integral[top, left]
        )
        window_square_sum = (
            square_integral[bottom, right]
            - square_integral[top, right]
            - square_integral[bottom, left]
            + square_integral[top, left]
        )
        window_count = (
            valid_integral[bottom, right]
            - valid_integral[top, right]
            - valid_integral[bottom, left]
            + valid_integral[top, left]
        )
        safe_count = np.where(window_count > 0, window_count, 1.0)
        mean = window_sum / safe_count
        mean_square = window_square_sum / safe_count
        row_variance = np.maximum(mean_square - (mean ** 2), 0.0)
        variance[row_index, valid_mask[row_index, :]] = row_variance[valid_mask[row_index, :]]
        next_heartbeat_at = _emit_throttled_heartbeat(
            heartbeat_callback,
            next_heartbeat_at,
            f"自动遮罩方差分析：正在处理第 {row_index + 1}/{height} 行。",
        )
        if progress_callback is not None:
            progress_callback(f"自动遮罩方差分析：已完成 {row_index + 1}/{height} 行。")

    valid_variance = variance[valid_mask]
    if valid_variance.size == 0:
        return np.zeros_like(height_array, dtype=np.float32)

    scale = float(np.percentile(valid_variance, 95))
    if scale <= 1e-6:
        scale = 1.0

    return 1.0 - np.clip(variance / scale, 0.0, 1.0)


def _label_connected_components_python(
    mask_array: np.ndarray,
    progress_callback: ProgressCallback | None = None,
    heartbeat_callback: ProgressCallback | None = None,
    heartbeat_label: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Label 8-connected true components without storing per-component cell lists."""

    height, width = mask_array.shape
    labels = np.full(mask_array.shape, -1, dtype=np.int32)
    component_sizes: list[int] = []
    next_heartbeat_at = monotonic() + 0.75
    component_counter = 0

    for row_index in range(height):
        for column_index in range(width):
            next_heartbeat_at = _emit_throttled_heartbeat(
                heartbeat_callback,
                next_heartbeat_at,
                (
                    heartbeat_label or "自动遮罩连通域标记"
                )
                + f"：正在扫描第 {row_index + 1}/{height} 行，已发现 {component_counter} 个目标连通域。",
            )
            if not mask_array[row_index, column_index]:
                continue
            if labels[row_index, column_index] >= 0:
                continue

            label_id = component_counter
            component_counter += 1
            component_size = 0
            pending_cells: deque[tuple[int, int]] = deque([(row_index, column_index)])
            labels[row_index, column_index] = label_id

            while pending_cells:
                current_row, current_column = pending_cells.popleft()
                component_size += 1
                if component_size == 1 or component_size % AUTO_MASK_COMPONENT_HEARTBEAT_INTERVAL == 0:
                    next_heartbeat_at = _emit_throttled_heartbeat(
                        heartbeat_callback,
                        next_heartbeat_at,
                        (
                            heartbeat_label or "自动遮罩连通域标记"
                        )
                        + (
                            f"：正在扩展第 {component_counter} 个目标连通域，"
                            f"已扫描 {component_size} 个像素，待处理队列 {len(pending_cells)}。"
                        ),
                    )

                for row_delta, column_delta in D8_OFFSETS:
                    neighbor_row = current_row + row_delta
                    neighbor_column = current_column + column_delta
                    if neighbor_row < 0 or neighbor_row >= height or neighbor_column < 0 or neighbor_column >= width:
                        continue
                    if not mask_array[neighbor_row, neighbor_column]:
                        continue
                    if labels[neighbor_row, neighbor_column] >= 0:
                        continue

                    labels[neighbor_row, neighbor_column] = label_id
                    pending_cells.append((neighbor_row, neighbor_column))

            component_sizes.append(component_size)
            next_heartbeat_at = _emit_throttled_heartbeat(
                heartbeat_callback,
                next_heartbeat_at,
                (
                    heartbeat_label or "自动遮罩连通域标记"
                )
                + f"：第 {component_counter} 个目标连通域扫描完成，大小 {component_size} 像素。",
            )

        if progress_callback is not None:
            progress_callback(f"{heartbeat_label or '自动遮罩连通域标记'}：已完成 {row_index + 1}/{height} 行。")

    return labels, np.asarray(component_sizes, dtype=np.int32)


def _label_connected_components(
    mask_array: np.ndarray,
    progress_callback: ProgressCallback | None = None,
    heartbeat_callback: ProgressCallback | None = None,
    heartbeat_label: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Label true components using Rust when available, else the Python fallback."""

    if rust_kernel_available():
        try:
            def _rust_callback(message: str) -> None:
                if message.startswith("ROW|"):
                    if progress_callback is not None:
                        progress_callback(message[4:])
                    return
                if message.startswith("HEARTBEAT|"):
                    if heartbeat_callback is not None:
                        heartbeat_callback(message[10:])
                    return
                if heartbeat_callback is not None:
                    heartbeat_callback(message)

            labels = label_connected_components_rust(mask_array, progress_callback=_rust_callback)
            valid_labels = labels[labels >= 0]
            if valid_labels.size == 0:
                return labels, np.asarray([], dtype=np.int32)
            component_sizes = np.bincount(valid_labels, minlength=int(valid_labels.max()) + 1).astype(np.int32)
            return labels, component_sizes
        except RuntimeError:
            pass

    return _label_connected_components_python(
        mask_array,
        progress_callback=progress_callback,
        heartbeat_callback=heartbeat_callback,
        heartbeat_label=heartbeat_label,
    )


def _build_border_connected_invalid_mask(
    candidate_mask: np.ndarray,
    valid_mask: np.ndarray,
    progress_callback: ProgressCallback | None = None,
    heartbeat_callback: ProgressCallback | None = None,
) -> np.ndarray:
    """Keep only invalid candidates that are connected to the raster border."""

    working_mask = candidate_mask & valid_mask
    if not working_mask.any():
        return np.zeros_like(candidate_mask, dtype=bool)

    labels, component_sizes = _label_connected_components(
        working_mask,
        progress_callback=progress_callback,
        heartbeat_callback=heartbeat_callback,
        heartbeat_label="自动遮罩边界连通分析",
    )
    if component_sizes.size == 0:
        return np.zeros_like(candidate_mask, dtype=bool)

    border_labels = np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
    border_labels = border_labels[border_labels >= 0]
    if border_labels.size == 0:
        return np.zeros_like(candidate_mask, dtype=bool)

    keep_lookup = np.zeros(component_sizes.shape[0], dtype=bool)
    keep_lookup[np.unique(border_labels)] = True
    connected_mask = np.zeros_like(candidate_mask, dtype=bool)
    valid_label_cells = labels >= 0
    connected_mask[valid_label_cells] = keep_lookup[labels[valid_label_cells]]
    return connected_mask


def _filter_small_components(
    mask_array: np.ndarray,
    *,
    keep_value: bool,
    min_component_size: int,
    progress_callback: ProgressCallback | None = None,
    heartbeat_callback: ProgressCallback | None = None,
    heartbeat_label: str = "",
) -> np.ndarray:
    """Remove or fill connected components smaller than the configured size."""

    target_mask = mask_array if keep_value else ~mask_array
    if not target_mask.any():
        return mask_array.copy()

    labels, component_sizes = _label_connected_components(
        target_mask,
        progress_callback=progress_callback,
        heartbeat_callback=heartbeat_callback,
        heartbeat_label=heartbeat_label or "自动遮罩连通域过滤",
    )
    if component_sizes.size == 0:
        return mask_array.copy()

    small_component_lookup = component_sizes < max(1, int(min_component_size))
    if not small_component_lookup.any():
        return mask_array.copy()

    filtered_mask = mask_array.copy()
    valid_label_cells = labels >= 0
    cells_to_flip = np.zeros_like(mask_array, dtype=bool)
    cells_to_flip[valid_label_cells] = small_component_lookup[labels[valid_label_cells]]
    filtered_mask[cells_to_flip] = not keep_value
    return filtered_mask


def _generate_auto_mask_core(
    height_array: np.ndarray,
    valid_mask: np.ndarray,
    border_sensitivity: float,
    texture_sensitivity: float,
    min_region_size: int,
    progress_callback: ProgressCallback | None = None,
    heartbeat_callback: ProgressCallback | None = None,
) -> np.ndarray:
    """Generate an auto mask on one raster resolution."""

    border_values = _collect_border_values(height_array, valid_mask)
    if border_values.size == 0:
        return valid_mask.copy()

    border_reference = float(np.median(border_values))
    border_scale = float(np.std(border_values))
    if border_scale <= 1e-6:
        border_scale = 1.0

    tolerance = max(1.0, border_scale * (2.25 / max(border_sensitivity, 0.1)))
    border_delta = np.abs(height_array - border_reference)
    border_similarity = np.clip(1.0 - (border_delta / tolerance), 0.0, 1.0).astype(np.float32)
    if progress_callback is not None:
        progress_callback("自动遮罩边界建模完成。")

    low_gradient_score = _compute_gradient_scores(
        height_array,
        valid_mask,
        progress_callback,
        heartbeat_callback,
    )
    low_variance_score = _compute_local_variance_scores(
        height_array,
        valid_mask,
        progress_callback,
        heartbeat_callback,
    )
    texture_weight = float(texture_sensitivity)
    raw_candidate_score = (
        0.45 * border_similarity
        + 0.30 * np.clip(low_gradient_score * texture_weight, 0.0, 1.0)
        + 0.25 * np.clip(low_variance_score * texture_weight, 0.0, 1.0)
    )
    invalid_candidate = valid_mask & (raw_candidate_score >= 0.58)
    connected_invalid = _build_border_connected_invalid_mask(
        invalid_candidate,
        valid_mask,
        progress_callback,
        heartbeat_callback,
    )
    auto_mask = valid_mask & ~connected_invalid
    auto_mask = _filter_small_components(
        auto_mask,
        keep_value=True,
        min_component_size=min_region_size,
        progress_callback=progress_callback,
        heartbeat_callback=heartbeat_callback,
        heartbeat_label="自动遮罩连通域过滤：正在移除过小有效区域。",
    )
    hole_fill_size = max(32, min_region_size // 2)
    auto_mask = _filter_small_components(
        auto_mask,
        keep_value=False,
        min_component_size=hole_fill_size,
        progress_callback=progress_callback,
        heartbeat_callback=heartbeat_callback,
        heartbeat_label="自动遮罩连通域过滤：正在填补过小空洞。",
    )
    return auto_mask


def generate_auto_mask(
    height_array: np.ndarray,
    valid_mask: np.ndarray,
    enabled: bool,
    border_sensitivity: float,
    texture_sensitivity: float,
    min_region_size: int,
    progress_callback: ProgressCallback | None = None,
    heartbeat_callback: ProgressCallback | None = None,
) -> np.ndarray:
    """
    Build one automatic validity mask from border/background, texture, and connectivity cues.

    The first version intentionally stays conservative: only border-connected regions
    that strongly resemble flat background are removed. Interior areas are preserved
    unless later manual or uploaded masks further restrict them.
    """

    if not enabled:
        return valid_mask.copy()

    analysis_scale = _choose_auto_mask_downsample_scale(int(height_array.shape[0]), int(height_array.shape[1]))
    if analysis_scale > 1:
        if heartbeat_callback is not None:
            heartbeat_callback(
                (
                    "自动遮罩降采样预分析："
                    f"原始尺寸 {height_array.shape[1]}x{height_array.shape[0]}，"
                    f"使用 {analysis_scale}x{analysis_scale} 分块降采样。"
                )
            )
        coarse_height, coarse_valid = _downsample_auto_mask_inputs(height_array, valid_mask, analysis_scale)
        coarse_min_region_size = _scale_component_threshold(min_region_size, analysis_scale)
        auto_mask = _generate_auto_mask_core(
            coarse_height,
            coarse_valid,
            border_sensitivity,
            texture_sensitivity,
            coarse_min_region_size,
            progress_callback=progress_callback,
            heartbeat_callback=heartbeat_callback,
        )
        auto_mask = _upsample_mask_to_shape(auto_mask, height_array.shape, analysis_scale) & valid_mask
        if progress_callback is not None:
            progress_callback(
                (
                    "自动遮罩降采样回投完成："
                    f"分析分辨率 {coarse_height.shape[1]}x{coarse_height.shape[0]}，"
                    f"已恢复到原始尺寸 {height_array.shape[1]}x{height_array.shape[0]}。"
                )
            )
    else:
        auto_mask = _generate_auto_mask_core(
            height_array,
            valid_mask,
            border_sensitivity,
            texture_sensitivity,
            min_region_size,
            progress_callback=progress_callback,
            heartbeat_callback=heartbeat_callback,
        )

    if progress_callback is not None:
        progress_callback(
            f"自动遮罩生成完成：保留 {int(auto_mask.sum())}/{int(valid_mask.sum())} 个有效像素。"
        )
    return auto_mask


def apply_box_smoothing(
    height_array: np.ndarray,
    valid_mask: np.ndarray,
    enabled: bool,
    kernel_size: int,
    progress_callback: ProgressCallback | None = None,
    heartbeat_callback: ProgressCallback | None = None,
) -> np.ndarray:
    """Apply a small box blur while keeping invalid pixels untouched."""

    if not enabled or kernel_size <= 1:
        return height_array

    radius = max((kernel_size - 1) // 2, 0)
    padded_height = np.pad(height_array, radius, mode="edge").astype(np.float32)
    padded_valid = np.pad(valid_mask.astype(np.float32), radius, mode="constant", constant_values=0.0)

    value_integral = np.pad(
        np.cumsum(np.cumsum(padded_height * padded_valid, axis=0), axis=1),
        ((1, 0), (1, 0)),
        mode="constant",
    )
    valid_integral = np.pad(
        np.cumsum(np.cumsum(padded_valid, axis=0), axis=1),
        ((1, 0), (1, 0)),
        mode="constant",
    )

    smoothed_array = height_array.copy().astype(np.float32)
    kernel_span = radius * 2 + 1
    height, width = height_array.shape
    next_heartbeat_at = monotonic() + 0.75

    for row_index in range(height):
        top = row_index
        bottom = row_index + kernel_span
        left = np.arange(width, dtype=np.int32)
        right = left + kernel_span

        window_sum = (
            value_integral[bottom, right]
            - value_integral[top, right]
            - value_integral[bottom, left]
            + value_integral[top, left]
        )
        window_count = (
            valid_integral[bottom, right]
            - valid_integral[top, right]
            - valid_integral[bottom, left]
            + valid_integral[top, left]
        )

        safe_count = np.where(window_count > 0, window_count, 1.0)
        row_average = window_sum / safe_count
        row_mask = valid_mask[row_index, :]
        smoothed_array[row_index, row_mask] = row_average[row_mask]
        next_heartbeat_at = _emit_throttled_heartbeat(
            heartbeat_callback,
            next_heartbeat_at,
            f"平滑卷积：正在处理第 {row_index + 1}/{height} 行，核大小 {kernel_size}。",
        )

        if progress_callback is not None:
            progress_callback(f"平滑卷积：已完成 {row_index + 1}/{height} 行，核大小 {kernel_size}。")

    return smoothed_array


def fill_local_sinks(
    height_array: np.ndarray,
    valid_mask: np.ndarray,
    enabled: bool,
    max_iterations: int = DEFAULT_FILL_SINK_MAX_ITERATIONS,
    progress_callback: ProgressCallback | None = None,
    heartbeat_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, int]:
    """
    Fill simple local depressions and repair closed flat basins.

    The first pass keeps the existing conservative local-minimum lifting rule.
    A second pass repairs equal-height closed flats that still have no outlet by
    raising them just enough to spill and imposing a tiny monotonic gradient
    toward one deterministic spill anchor.
    """

    if not enabled:
        return height_array, 0

    filled_array = height_array.copy()
    iterations_used = 0
    for iteration in range(max_iterations):
        if heartbeat_callback is not None:
            heartbeat_callback(f"填洼迭代 {iteration + 1}/{max_iterations}：开始执行。")
        updated_array, changed_cells = fill_sink_iteration(
            filled_array,
            valid_mask,
            progress_callback=progress_callback,
        )
        repaired_array, repaired_cells = repair_closed_flat_basins(
            updated_array,
            valid_mask,
            progress_callback=progress_callback,
            heartbeat_callback=heartbeat_callback,
            iteration_index=iteration + 1,
            total_iterations=max_iterations,
        )
        filled_array = repaired_array
        total_changed_cells = changed_cells + repaired_cells
        if progress_callback is not None:
            progress_callback(
                f"填洼迭代 {iteration + 1}/{max_iterations}：局部修复 {changed_cells}，封闭平坡修复 {repaired_cells}。"
            )
        if total_changed_cells == 0:
            break
        iterations_used = iteration + 1

    return filled_array, iterations_used


def fill_sink_iteration(
    height_array: np.ndarray,
    valid_mask: np.ndarray,
    progress_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, int]:
    """
    Execute one conservative sink-fill pass on the raster interior.

    This pass used to rely on nested Python loops over every pixel. That made
    the task thread hold the GIL for too long, which in turn starved the API
    polling thread and looked like a frontend freeze. The vectorized version
    keeps the rule identical while moving the heavy work into NumPy.
    """

    updated_array = height_array.copy()
    height, width = height_array.shape
    if height < 3 or width < 3:
        return updated_array, 0

    if progress_callback is not None:
        progress_callback("局部填洼：开始汇总 8 邻域最低高程。")

    padded_height = np.pad(height_array, 1, mode="constant", constant_values=np.inf).astype(np.float32)
    padded_valid = np.pad(valid_mask, 1, mode="constant", constant_values=False)
    interior_shape = (height - 2, width - 2)
    minimum_neighbor = np.full(interior_shape, np.inf, dtype=np.float32)
    neighbor_sum = np.zeros(interior_shape, dtype=np.float32)
    neighbor_count = np.zeros(interior_shape, dtype=np.int16)
    equal_height_neighbor_count = np.zeros(interior_shape, dtype=np.int16)
    interior_values = height_array[1:-1, 1:-1]

    for direction_index, (row_delta, column_delta) in enumerate(D8_OFFSETS, start=1):
        row_start = 2 + row_delta
        column_start = 2 + column_delta
        neighbor_values = padded_height[row_start : row_start + height - 2, column_start : column_start + width - 2]
        neighbor_valid = padded_valid[row_start : row_start + height - 2, column_start : column_start + width - 2]
        safe_neighbor_values = np.where(neighbor_valid, neighbor_values, np.inf)
        minimum_neighbor = np.minimum(minimum_neighbor, safe_neighbor_values)
        neighbor_sum += np.where(neighbor_valid, neighbor_values, 0.0).astype(np.float32)
        neighbor_count += neighbor_valid.astype(np.int16)
        equal_height_neighbor_count += (
            neighbor_valid
            & np.isclose(neighbor_values, interior_values)
        ).astype(np.int16)
        if progress_callback is not None:
            progress_callback(f"局部填洼邻域汇总：已完成 {direction_index}/8 个方向。")

    interior_valid = valid_mask[1:-1, 1:-1]
    change_mask = interior_valid & np.isfinite(minimum_neighbor) & (interior_values < minimum_neighbor)
    isolated_sink_mask = change_mask & (equal_height_neighbor_count == 0) & (neighbor_count > 0)
    average_neighbor = np.divide(
        neighbor_sum,
        np.maximum(neighbor_count, 1),
        dtype=np.float32,
    )
    fill_values = np.where(isolated_sink_mask, average_neighbor, minimum_neighbor)
    updated_array[1:-1, 1:-1] = np.where(change_mask, fill_values, interior_values)
    changed_cells = int(change_mask.sum())

    if progress_callback is not None:
        progress_callback(
            "局部填洼更新完成："
            f"本轮修复 {changed_cells} 个像素，其中单像素洼地 {int(isolated_sink_mask.sum())} 个。"
        )

    return updated_array, changed_cells


def repair_closed_flat_basins(
    height_array: np.ndarray,
    valid_mask: np.ndarray,
    progress_callback: ProgressCallback | None = None,
    heartbeat_callback: ProgressCallback | None = None,
    iteration_index: int = 1,
    total_iterations: int = 1,
) -> tuple[np.ndarray, int]:
    """
    Repair flat regions that have no lower outlet.

    Each closed equal-height basin is lifted to the minimum spill elevation
    found around its boundary and given a tiny increasing gradient away from
    one deterministic spill anchor. This keeps the modification minimal while
    ensuring later flow routing can leave the basin.
    """

    repaired_array = height_array.copy()
    visited = np.zeros_like(valid_mask, dtype=bool)
    changed_cells = 0
    epsilon = np.float32(1e-3)
    height, width = height_array.shape
    next_heartbeat_at = monotonic() + 0.75

    for row_index in range(height):
        for column_index in range(width):
            if visited[row_index, column_index]:
                continue
            if not valid_mask[row_index, column_index]:
                continue

            region_cells = collect_equal_height_region(
                row_index,
                column_index,
                height_array,
                valid_mask,
                visited,
                heartbeat_callback=heartbeat_callback,
                heartbeat_label=(
                    f"填洼迭代 {iteration_index}/{total_iterations}："
                    f"封闭平坡修复正在扩展第 {row_index + 1} 行附近的等高区。"
                ),
            )
            if len(region_cells) <= 1:
                continue

            next_heartbeat_at = _emit_throttled_heartbeat(
                heartbeat_callback,
                next_heartbeat_at,
                (
                    f"填洼迭代 {iteration_index}/{total_iterations}："
                    f"正在分析大小为 {len(region_cells)} 的封闭平坡区出口。"
                ),
            )
            outlet_info = analyze_flat_region_outlets(
                region_cells,
                height_array,
                valid_mask,
                heartbeat_callback=heartbeat_callback,
                heartbeat_label=(
                    f"填洼迭代 {iteration_index}/{total_iterations}："
                    f"正在扫描大小为 {len(region_cells)} 的封闭平坡区边界。"
                ),
            )
            if outlet_info["has_lower_outlet"]:
                continue

            spill_elevation = outlet_info["spill_elevation"]
            spill_anchor = outlet_info["spill_anchor"]
            if spill_elevation is None or spill_anchor is None:
                continue

            repaired_array = impose_closed_flat_gradient(
                repaired_array,
                region_cells,
                spill_anchor,
                float(spill_elevation),
                epsilon,
                heartbeat_callback=heartbeat_callback,
                heartbeat_label=(
                    f"填洼迭代 {iteration_index}/{total_iterations}："
                    f"正在给大小为 {len(region_cells)} 的封闭平坡区施加微坡度。"
                ),
            )
            changed_cells += len(region_cells)

        if progress_callback is not None:
            progress_callback(f"封闭平坡修复：已完成 {row_index + 1}/{height} 行。")
        next_heartbeat_at = _emit_throttled_heartbeat(
            heartbeat_callback,
            next_heartbeat_at,
            (
                f"填洼迭代 {iteration_index}/{total_iterations}："
                f"封闭平坡修复已完成 {row_index + 1}/{height} 行。"
            ),
        )

    return repaired_array, changed_cells


def collect_equal_height_region(
    start_row: int,
    start_column: int,
    height_array: np.ndarray,
    valid_mask: np.ndarray,
    visited: np.ndarray,
    heartbeat_callback: ProgressCallback | None = None,
    heartbeat_label: str = "",
) -> list[tuple[int, int]]:
    """Collect one 8-connected equal-height region inside the valid mask."""

    target_height = float(height_array[start_row, start_column])
    region_cells: list[tuple[int, int]] = []
    pending_cells: deque[tuple[int, int]] = deque([(start_row, start_column)])
    visited[start_row, start_column] = True
    height, width = height_array.shape
    next_heartbeat_at = monotonic() + 0.75

    while pending_cells:
        row_index, column_index = pending_cells.popleft()
        region_cells.append((row_index, column_index))
        next_heartbeat_at = _emit_throttled_heartbeat(
            heartbeat_callback,
            next_heartbeat_at,
            heartbeat_label or f"封闭平坡修复：正在扩展等高区，已收集 {len(region_cells)} 个像素。",
        )

        for row_delta, column_delta in D8_OFFSETS:
            neighbor_row = row_index + row_delta
            neighbor_column = column_index + column_delta
            if neighbor_row < 0 or neighbor_row >= height or neighbor_column < 0 or neighbor_column >= width:
                continue
            if visited[neighbor_row, neighbor_column]:
                continue
            if not valid_mask[neighbor_row, neighbor_column]:
                continue
            if not np.isclose(float(height_array[neighbor_row, neighbor_column]), target_height):
                continue

            visited[neighbor_row, neighbor_column] = True
            pending_cells.append((neighbor_row, neighbor_column))

    return region_cells


def analyze_flat_region_outlets(
    region_cells: list[tuple[int, int]],
    height_array: np.ndarray,
    valid_mask: np.ndarray,
    heartbeat_callback: ProgressCallback | None = None,
    heartbeat_label: str = "",
) -> dict[str, object]:
    """Inspect one equal-height region for real outlets and spill candidates."""

    region_height = float(height_array[region_cells[0][0], region_cells[0][1]])
    region_set = set(region_cells)
    spill_elevation: float | None = None
    spill_anchor: tuple[int, int] | None = None
    has_lower_outlet = False
    height, width = height_array.shape
    next_heartbeat_at = monotonic() + 0.75

    for row_index, column_index in region_cells:
        next_heartbeat_at = _emit_throttled_heartbeat(
            heartbeat_callback,
            next_heartbeat_at,
            heartbeat_label or f"封闭平坡修复：正在扫描等高区边界，区域大小 {len(region_cells)}。",
        )
        for row_delta, column_delta in D8_OFFSETS:
            neighbor_row = row_index + row_delta
            neighbor_column = column_index + column_delta
            neighbor_cell = (neighbor_row, neighbor_column)
            if neighbor_row < 0 or neighbor_row >= height or neighbor_column < 0 or neighbor_column >= width:
                continue
            if neighbor_cell in region_set:
                continue
            if not valid_mask[neighbor_row, neighbor_column]:
                continue

            neighbor_height = float(height_array[neighbor_row, neighbor_column])
            if neighbor_height < region_height - 1e-6:
                has_lower_outlet = True
                continue

            if spill_elevation is None or neighbor_height < spill_elevation - 1e-6:
                spill_elevation = neighbor_height
                spill_anchor = (row_index, column_index)
                continue

            if spill_elevation is not None and abs(neighbor_height - spill_elevation) <= 1e-6:
                if spill_anchor is None or (row_index, column_index) < spill_anchor:
                    spill_anchor = (row_index, column_index)

    return {
        "has_lower_outlet": has_lower_outlet,
        "spill_elevation": spill_elevation,
        "spill_anchor": spill_anchor,
    }


def impose_closed_flat_gradient(
    height_array: np.ndarray,
    region_cells: list[tuple[int, int]],
    spill_anchor: tuple[int, int],
    spill_elevation: float,
    epsilon: np.float32,
    heartbeat_callback: ProgressCallback | None = None,
    heartbeat_label: str = "",
) -> np.ndarray:
    """Raise one closed flat basin and add a tiny monotonic slope to its spill anchor."""

    updated_array = height_array.copy()
    region_set = set(region_cells)
    distance_map: dict[tuple[int, int], int] = {spill_anchor: 0}
    pending_cells: deque[tuple[int, int]] = deque([spill_anchor])
    height, width = height_array.shape
    next_heartbeat_at = monotonic() + 0.75

    while pending_cells:
        row_index, column_index = pending_cells.popleft()
        current_distance = distance_map[(row_index, column_index)]
        next_heartbeat_at = _emit_throttled_heartbeat(
            heartbeat_callback,
            next_heartbeat_at,
            heartbeat_label or f"封闭平坡修复：正在传播微坡度，已覆盖 {len(distance_map)} 个像素。",
        )

        for row_delta, column_delta in D8_OFFSETS:
            neighbor_row = row_index + row_delta
            neighbor_column = column_index + column_delta
            neighbor_cell = (neighbor_row, neighbor_column)
            if neighbor_row < 0 or neighbor_row >= height or neighbor_column < 0 or neighbor_column >= width:
                continue
            if neighbor_cell not in region_set:
                continue
            if neighbor_cell in distance_map:
                continue

            distance_map[neighbor_cell] = current_distance + 1
            pending_cells.append(neighbor_cell)

    for row_index, column_index in region_cells:
        distance_to_anchor = distance_map.get((row_index, column_index), 0)
        updated_array[row_index, column_index] = np.float32(
            spill_elevation + float(epsilon) * float(distance_to_anchor + 1)
        )

    return updated_array


def terrain_preview_image(
    height_array: np.ndarray,
    valid_mask: np.ndarray,
) -> Image.Image:
    """Convert a processed terrain raster into a preview image."""

    if valid_mask.any():
        valid_values = height_array[valid_mask]
        minimum_value = float(valid_values.min())
        maximum_value = float(valid_values.max())
    else:
        minimum_value = 0.0
        maximum_value = 255.0

    value_range = maximum_value - minimum_value
    if value_range <= 0:
        normalized = np.zeros_like(height_array, dtype=np.uint8)
    else:
        normalized = ((height_array - minimum_value) / value_range * 255.0).clip(0, 255).astype(np.uint8)

    preview_array = np.where(valid_mask, normalized, 0)
    return Image.fromarray(preview_array, mode="L")


def terrain_statistics_message(
    height_array: np.ndarray,
    valid_mask: np.ndarray,
) -> str:
    """Build a concise summary of one processed terrain raster."""

    valid_values = height_array[valid_mask] if valid_mask.any() else height_array.reshape(-1)
    return (
        "Terrain statistics: "
        f"shape={height_array.shape}, min={float(valid_values.min()):.2f}, "
        f"max={float(valid_values.max()):.2f}, mean={float(valid_values.mean()):.2f}."
    )


def _build_neighbor_lookup(
    height: int,
    width: int,
    heartbeat_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-compute valid neighbor coordinates for each direction."""

    neighbor_rows = np.full((8, height, width), -1, dtype=np.int32)
    neighbor_columns = np.full((8, height, width), -1, dtype=np.int32)
    next_heartbeat_at = monotonic() + 0.75

    for direction_index, (row_delta, column_delta) in enumerate(D8_OFFSETS):
        for row_index in range(height):
            for column_index in range(width):
                neighbor_row = row_index + row_delta
                neighbor_column = column_index + column_delta
                if neighbor_row < 0 or neighbor_row >= height or neighbor_column < 0 or neighbor_column >= width:
                    continue

                neighbor_rows[direction_index, row_index, column_index] = neighbor_row
                neighbor_columns[direction_index, row_index, column_index] = neighbor_column
                next_heartbeat_at = _emit_throttled_heartbeat(
                    heartbeat_callback,
                    next_heartbeat_at,
                    (
                        "D8 1/4 严格坡降：邻域索引预构建，"
                        f"方向 {direction_index + 1}/8，第 {row_index + 1}/{height} 行。"
                    ),
                )

    return neighbor_rows, neighbor_columns


def _compute_strict_d8_pass(
    height_array: np.ndarray,
    valid_mask: np.ndarray,
    neighbor_rows: np.ndarray,
    neighbor_columns: np.ndarray,
    slope_weight: float,
    progress_callback: ProgressCallback | None = None,
    heartbeat_callback: ProgressCallback | None = None,
) -> np.ndarray:
    """Assign directions only when a strictly lower neighbor exists."""

    height, width = height_array.shape
    direction_array = np.full((height, width), -1, dtype=np.int8)
    next_heartbeat_at = monotonic() + 0.75

    for row_index in range(height):
        for column_index in range(width):
            if not valid_mask[row_index, column_index]:
                continue
            next_heartbeat_at = _emit_throttled_heartbeat(
                heartbeat_callback,
                next_heartbeat_at,
                f"D8 1/4 严格坡降：正在扫描第 {row_index + 1}/{height} 行。",
            )
            current_height = float(height_array[row_index, column_index])
            raw_slopes = np.zeros(8, dtype=np.float32)

            for direction_index in range(8):
                neighbor_row = int(neighbor_rows[direction_index, row_index, column_index])
                neighbor_column = int(neighbor_columns[direction_index, row_index, column_index])
                if neighbor_row < 0 or neighbor_column < 0:
                    continue
                if not valid_mask[neighbor_row, neighbor_column]:
                    continue

                neighbor_height = float(height_array[neighbor_row, neighbor_column])
                height_drop = current_height - neighbor_height
                if height_drop <= 0:
                    continue

                raw_slopes[direction_index] = height_drop / float(D8_DISTANCES[direction_index])

            direction_array[row_index, column_index] = _select_best_scored_direction(
                slope_scores=_normalize_scores(raw_slopes),
                flat_escape_scores=np.zeros(8, dtype=np.float32),
                outlet_proximity_scores=np.zeros(8, dtype=np.float32),
                continuity_scores=np.zeros(8, dtype=np.float32),
                slope_weight=slope_weight,
                flat_escape_weight=0.0,
                outlet_proximity_weight=0.0,
                continuity_weight=0.0,
            )

        if progress_callback is not None:
            progress_callback(f"D8 1/4 严格坡降：已完成 {row_index + 1}/{height} 行。")

    return direction_array


def _collect_flat_region(
    start_row: int,
    start_column: int,
    height_array: np.ndarray,
    valid_mask: np.ndarray,
    visited: np.ndarray,
    neighbor_rows: np.ndarray,
    neighbor_columns: np.ndarray,
) -> list[tuple[int, int]]:
    """Collect one full 8-connected equal-height region."""

    target_height = float(height_array[start_row, start_column])
    region_cells: list[tuple[int, int]] = []
    pending_cells: deque[tuple[int, int]] = deque([(start_row, start_column)])
    visited[start_row, start_column] = True

    while pending_cells:
        row_index, column_index = pending_cells.popleft()
        region_cells.append((row_index, column_index))

        for direction_index in range(8):
            neighbor_row = int(neighbor_rows[direction_index, row_index, column_index])
            neighbor_column = int(neighbor_columns[direction_index, row_index, column_index])
            if neighbor_row < 0 or neighbor_column < 0:
                continue
            if visited[neighbor_row, neighbor_column]:
                continue
            if not valid_mask[neighbor_row, neighbor_column]:
                continue
            if not np.isclose(float(height_array[neighbor_row, neighbor_column]), target_height):
                continue

            visited[neighbor_row, neighbor_column] = True
            pending_cells.append((neighbor_row, neighbor_column))

    return region_cells


def _build_flat_region_labels_python(
    height_array: np.ndarray,
    valid_mask: np.ndarray,
    neighbor_rows: np.ndarray,
    neighbor_columns: np.ndarray,
    heartbeat_callback: ProgressCallback | None = None,
) -> np.ndarray:
    """Build equal-height region labels with the Python BFS fallback."""

    height, width = height_array.shape
    visited = np.zeros((height, width), dtype=bool)
    label_array = np.full((height, width), -1, dtype=np.int32)
    current_label = 0
    next_heartbeat_at = monotonic() + 0.75

    for row_index in range(height):
        for column_index in range(width):
            next_heartbeat_at = _emit_throttled_heartbeat(
                heartbeat_callback,
                next_heartbeat_at,
                (
                    "D8 2/4 平坡导流：平坡分区标记，"
                    f"正在扫描第 {row_index + 1}/{height} 行，已发现 {current_label} 个平坡区。"
                ),
            )
            if visited[row_index, column_index]:
                continue
            if not valid_mask[row_index, column_index]:
                continue

            region_cells = _collect_flat_region(
                row_index,
                column_index,
                height_array,
                valid_mask,
                visited,
                neighbor_rows,
                neighbor_columns,
            )
            for region_row, region_column in region_cells:
                label_array[region_row, region_column] = current_label
            current_label += 1

    return label_array


def _group_region_cells_by_label(
    region_labels: np.ndarray,
) -> dict[int, list[tuple[int, int]]]:
    """Convert one label raster into explicit cell lists for each flat region."""

    region_groups: dict[int, list[tuple[int, int]]] = {}
    rows, columns = np.nonzero(region_labels >= 0)
    for row_index, column_index in zip(rows.tolist(), columns.tolist()):
        label_value = int(region_labels[row_index, column_index])
        region_groups.setdefault(label_value, []).append((row_index, column_index))

    return region_groups


def _group_outlet_segments(
    outlet_candidates: list[tuple[int, int, float]],
    neighbor_rows: np.ndarray,
    neighbor_columns: np.ndarray,
) -> list[list[tuple[int, int, float]]]:
    """Merge adjacent outlet boundary points into outlet segments."""

    if not outlet_candidates:
        return []

    candidate_positions = {(row_index, column_index) for row_index, column_index, _ in outlet_candidates}
    candidate_strengths = {
        (row_index, column_index): float(height_drop)
        for row_index, column_index, height_drop in outlet_candidates
    }
    visited: set[tuple[int, int]] = set()
    segments: list[list[tuple[int, int, float]]] = []

    for row_index, column_index, _ in outlet_candidates:
        start_cell = (row_index, column_index)
        if start_cell in visited:
            continue

        pending_cells: deque[tuple[int, int]] = deque([start_cell])
        visited.add(start_cell)
        segment_cells: list[tuple[int, int, float]] = []

        while pending_cells:
            current_row, current_column = pending_cells.popleft()
            segment_cells.append(
                (current_row, current_column, candidate_strengths[(current_row, current_column)])
            )

            for direction_index in range(8):
                neighbor_row = int(neighbor_rows[direction_index, current_row, current_column])
                neighbor_column = int(neighbor_columns[direction_index, current_row, current_column])
                neighbor_cell = (neighbor_row, neighbor_column)
                if neighbor_row < 0 or neighbor_column < 0:
                    continue
                if neighbor_cell not in candidate_positions:
                    continue
                if neighbor_cell in visited:
                    continue

                visited.add(neighbor_cell)
                pending_cells.append(neighbor_cell)

        segments.append(segment_cells)

    return segments


def _normalize_scores(raw_scores: np.ndarray) -> np.ndarray:
    """Normalize positive candidate scores into the `[0, 1]` range."""

    max_score = float(raw_scores.max(initial=0.0))
    if max_score <= 0:
        return np.zeros_like(raw_scores, dtype=np.float32)

    return (raw_scores / max_score).astype(np.float32)


def _select_best_scored_direction(
    slope_scores: np.ndarray,
    flat_escape_scores: np.ndarray,
    outlet_proximity_scores: np.ndarray,
    continuity_scores: np.ndarray,
    slope_weight: float,
    flat_escape_weight: float,
    outlet_proximity_weight: float,
    continuity_weight: float,
) -> int:
    """Choose the highest-scoring direction with deterministic tie breaking."""

    composite_scores = (
        slope_scores * float(slope_weight)
        + flat_escape_scores * float(flat_escape_weight)
        + outlet_proximity_scores * float(outlet_proximity_weight)
        + continuity_scores * float(continuity_weight)
    )

    best_direction = -1
    best_score = 0.0
    for direction_index in range(8):
        score = float(composite_scores[direction_index])
        if score <= 0:
            continue
        if score > best_score:
            best_score = score
            best_direction = direction_index

    return best_direction


def _compute_segment_strength_with_weight(
    segment_cells: list[tuple[int, int, float]],
    length_weight: float,
) -> float:
    """Score one outlet segment using height drop and segment length."""

    drops = [height_drop for _, _, height_drop in segment_cells]
    mean_drop = float(np.mean(drops)) if drops else 0.0
    segment_length = len(segment_cells)
    return mean_drop + length_weight * float(np.log1p(segment_length))


def _compute_segment_center(segment_cells: list[tuple[int, int, float]]) -> tuple[float, float]:
    """Return the geometric center of one outlet segment."""

    row_values = [float(row_index) for row_index, _, _ in segment_cells]
    column_values = [float(column_index) for _, column_index, _ in segment_cells]
    return float(np.mean(row_values)), float(np.mean(column_values))


def _extract_outlet_segments(
    region_cells: list[tuple[int, int]],
    height_array: np.ndarray,
    valid_mask: np.ndarray,
    neighbor_rows: np.ndarray,
    neighbor_columns: np.ndarray,
    flat_outlet_length_weight: float,
    heartbeat_callback: ProgressCallback | None = None,
    outlet_drop_map: np.ndarray | None = None,
) -> list[dict[str, object]]:
    """Find grouped outlet segments for one flat region."""

    region_lookup = set(region_cells)
    outlet_candidates: list[tuple[int, int, float]] = []
    next_heartbeat_at = monotonic() + 0.75

    for scanned_cells, (row_index, column_index) in enumerate(region_cells, start=1):
        next_heartbeat_at = _emit_throttled_heartbeat(
            heartbeat_callback,
            next_heartbeat_at,
            f"D8 2/4 平坡导流：正在搜索出口，已扫描 {scanned_cells}/{len(region_cells)} 个像素。",
        )
        if outlet_drop_map is not None:
            strongest_drop = float(outlet_drop_map[row_index, column_index])
        else:
            current_height = float(height_array[row_index, column_index])
            strongest_drop = 0.0

            for direction_index in range(8):
                neighbor_row = int(neighbor_rows[direction_index, row_index, column_index])
                neighbor_column = int(neighbor_columns[direction_index, row_index, column_index])
                if neighbor_row < 0 or neighbor_column < 0:
                    continue
                if not valid_mask[neighbor_row, neighbor_column]:
                    continue
                if (neighbor_row, neighbor_column) in region_lookup:
                    continue

                neighbor_height = float(height_array[neighbor_row, neighbor_column])
                height_drop = current_height - neighbor_height
                if height_drop <= 0:
                    continue

                strongest_drop = max(strongest_drop, height_drop)

        if strongest_drop > 0:
            outlet_candidates.append((row_index, column_index, strongest_drop))

    segments = _group_outlet_segments(outlet_candidates, neighbor_rows, neighbor_columns)
    outlet_segments: list[dict[str, object]] = []
    for segment_index, segment_cells in enumerate(segments):
        outlet_segments.append(
            {
                "segment_index": segment_index,
                "cells": segment_cells,
                "strength": _compute_segment_strength_with_weight(
                    segment_cells,
                    flat_outlet_length_weight,
                ),
                "center": _compute_segment_center(segment_cells),
            }
        )

    return outlet_segments


def _assign_flat_region_segments(
    region_cells: list[tuple[int, int]],
    outlet_segments: list[dict[str, object]],
    neighbor_rows: np.ndarray,
    neighbor_columns: np.ndarray,
    flat_outlet_distance_weight: float,
    heartbeat_callback: ProgressCallback | None = None,
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    """Assign each flat-region cell to the most suitable outlet segment."""

    assignment: dict[tuple[int, int], int] = {}
    distance_map: dict[tuple[int, int], int] = {}
    best_cost: dict[tuple[int, int], float] = {}
    region_set = set(region_cells)
    pending_cells: deque[tuple[int, int, int]] = deque()
    next_heartbeat_at = monotonic() + 0.75

    for segment in outlet_segments:
        segment_index = int(segment["segment_index"])
        segment_strength = float(segment["strength"])
        for row_index, column_index, _ in segment["cells"]:
            cell = (row_index, column_index)
            candidate_cost = -flat_outlet_distance_weight * segment_strength
            previous_cost = best_cost.get(cell)
            if previous_cost is not None and candidate_cost >= previous_cost:
                continue

            best_cost[cell] = candidate_cost
            assignment[cell] = segment_index
            distance_map[cell] = 0
            pending_cells.append((row_index, column_index, segment_index))

    while pending_cells:
        row_index, column_index, segment_index = pending_cells.popleft()
        current_distance = distance_map[(row_index, column_index)]
        segment_strength = float(outlet_segments[segment_index]["strength"])
        next_heartbeat_at = _emit_throttled_heartbeat(
            heartbeat_callback,
            next_heartbeat_at,
            f"D8 2/4 平坡导流：正在扩散出口归属，待处理队列 {len(pending_cells)}。",
        )

        for direction_index in range(8):
            neighbor_row = int(neighbor_rows[direction_index, row_index, column_index])
            neighbor_column = int(neighbor_columns[direction_index, row_index, column_index])
            neighbor_cell = (neighbor_row, neighbor_column)
            if neighbor_row < 0 or neighbor_column < 0:
                continue
            if neighbor_cell not in region_set:
                continue

            candidate_distance = current_distance + 1
            candidate_cost = candidate_distance - flat_outlet_distance_weight * segment_strength
            previous_cost = best_cost.get(neighbor_cell)
            if previous_cost is None or candidate_cost < previous_cost - 1e-6:
                best_cost[neighbor_cell] = candidate_cost
                assignment[neighbor_cell] = segment_index
                distance_map[neighbor_cell] = candidate_distance
                pending_cells.append((neighbor_row, neighbor_column, segment_index))
                continue

            if abs(candidate_cost - previous_cost) <= 1e-6:
                previous_segment = assignment[neighbor_cell]
                if segment_index < previous_segment:
                    assignment[neighbor_cell] = segment_index
                    distance_map[neighbor_cell] = candidate_distance
                    pending_cells.append((neighbor_row, neighbor_column, segment_index))

    return assignment, distance_map


def _choose_flat_neighbor_direction(
    row_index: int,
    column_index: int,
    segment_index: int,
    assignment: dict[tuple[int, int], int],
    distance_map: dict[tuple[int, int], int],
    outlet_segments: list[dict[str, object]],
    neighbor_rows: np.ndarray,
    neighbor_columns: np.ndarray,
    slope_weight: float,
    flat_escape_weight: float,
    outlet_proximity_weight: float,
    continuity_weight: float,
) -> int:
    """Pick a same-region neighbor via weighted flat-routing scores."""

    current_distance = distance_map[(row_index, column_index)]
    if current_distance <= 0:
        return -1

    flat_escape_scores = np.zeros(8, dtype=np.float32)
    outlet_proximity_raw = np.zeros(8, dtype=np.float32)
    continuity_raw = np.zeros(8, dtype=np.float32)
    segment_strength = float(outlet_segments[segment_index]["strength"])
    target_row, target_column = outlet_segments[segment_index]["center"]

    for direction_index in range(8):
        neighbor_row = int(neighbor_rows[direction_index, row_index, column_index])
        neighbor_column = int(neighbor_columns[direction_index, row_index, column_index])
        neighbor_cell = (neighbor_row, neighbor_column)
        if neighbor_row < 0 or neighbor_column < 0:
            continue
        if assignment.get(neighbor_cell) != segment_index:
            continue

        neighbor_distance = distance_map.get(neighbor_cell)
        if neighbor_distance is None or neighbor_distance >= current_distance:
            continue

        flat_escape_scores[direction_index] = (current_distance - neighbor_distance) / max(current_distance, 1)
        outlet_proximity_raw[direction_index] = segment_strength / (1.0 + float(neighbor_distance))
        candidate_row_delta = float(neighbor_row - row_index)
        candidate_column_delta = float(neighbor_column - column_index)
        target_row_delta = float(target_row - row_index)
        target_column_delta = float(target_column - column_index)
        target_norm = float(np.hypot(target_row_delta, target_column_delta))
        candidate_norm = float(np.hypot(candidate_row_delta, candidate_column_delta))
        if target_norm > 0.0 and candidate_norm > 0.0:
            alignment = (
                candidate_row_delta * target_row_delta + candidate_column_delta * target_column_delta
            ) / (candidate_norm * target_norm)
            continuity_raw[direction_index] = max(0.0, alignment)

    return _select_best_scored_direction(
        slope_scores=np.zeros(8, dtype=np.float32),
        flat_escape_scores=flat_escape_scores,
        outlet_proximity_scores=_normalize_scores(outlet_proximity_raw),
        continuity_scores=_normalize_scores(continuity_raw),
        slope_weight=slope_weight,
        flat_escape_weight=flat_escape_weight,
        outlet_proximity_weight=outlet_proximity_weight,
        continuity_weight=continuity_weight,
    )


def _resolve_flat_regions(
    height_array: np.ndarray,
    valid_mask: np.ndarray,
    direction_array: np.ndarray,
    neighbor_rows: np.ndarray,
    neighbor_columns: np.ndarray,
    slope_weight: float,
    flat_escape_weight: float,
    outlet_proximity_weight: float,
    continuity_weight: float,
    flat_outlet_length_weight: float,
    flat_outlet_distance_weight: float,
    progress_callback: ProgressCallback | None = None,
    heartbeat_callback: ProgressCallback | None = None,
    use_rust_kernel: bool = False,
) -> np.ndarray:
    """Assign flow directions inside flat regions that have real downstream outlets."""

    resolved_directions = direction_array.copy()
    outlet_drop_map: np.ndarray | None = None
    if use_rust_kernel and rust_kernel_available():
        region_labels = label_equal_height_regions_rust(
            height_array,
            valid_mask,
            progress_callback=heartbeat_callback,
        )
        outlet_drop_map = compute_flat_outlet_drop_map_rust(
            height_array,
            valid_mask,
            region_labels,
            progress_callback=heartbeat_callback,
        )
    else:
        region_labels = _build_flat_region_labels_python(
            height_array,
            valid_mask,
            neighbor_rows,
            neighbor_columns,
            heartbeat_callback=heartbeat_callback,
        )

    region_groups = _group_region_cells_by_label(region_labels)
    region_items = sorted(region_groups.items(), key=lambda item: item[0])
    total_regions = len(region_items)
    next_heartbeat_at = monotonic() + 0.75

    for region_offset, (_, region_cells) in enumerate(region_items, start=1):
        if heartbeat_callback is not None and monotonic() >= next_heartbeat_at:
            heartbeat_callback(
                f"D8 2/4 平坡导流：正在处理第 {region_offset}/{max(total_regions, 1)} 个平坡区。"
            )
            next_heartbeat_at = monotonic() + 0.75
        if len(region_cells) <= 1:
            continue
        if not any(int(direction_array[row_index, column_index]) == -1 for row_index, column_index in region_cells):
            continue

        outlet_segments = _extract_outlet_segments(
            region_cells,
            height_array,
            valid_mask,
            neighbor_rows,
            neighbor_columns,
            flat_outlet_length_weight,
            heartbeat_callback=heartbeat_callback,
            outlet_drop_map=outlet_drop_map,
        )
        if not outlet_segments:
            continue

        assignment, distance_map = _assign_flat_region_segments(
            region_cells,
            outlet_segments,
            neighbor_rows,
            neighbor_columns,
            flat_outlet_distance_weight,
            heartbeat_callback=heartbeat_callback,
        )

        total_region_cells = len(region_cells)
        for cell_offset, (region_row, region_column) in enumerate(region_cells, start=1):
            if heartbeat_callback is not None and monotonic() >= next_heartbeat_at:
                heartbeat_callback(
                    "D8 2/4 平坡导流："
                    f"第 {region_offset}/{max(total_regions, 1)} 个平坡区，"
                    f"已扫描 {cell_offset}/{total_region_cells} 个像素。"
                )
                next_heartbeat_at = monotonic() + 0.75
            if int(direction_array[region_row, region_column]) != -1:
                continue

            chosen_segment = assignment.get((region_row, region_column))
            if chosen_segment is None:
                continue

            best_direction = _choose_flat_neighbor_direction(
                region_row,
                region_column,
                chosen_segment,
                assignment,
                distance_map,
                outlet_segments,
                neighbor_rows,
                neighbor_columns,
                slope_weight,
                flat_escape_weight,
                outlet_proximity_weight,
                continuity_weight,
            )
            if best_direction >= 0:
                resolved_directions[region_row, region_column] = best_direction

        if progress_callback is not None:
            progress_callback(f"D8 2/4 平坡导流：已完成 {region_offset}/{max(total_regions, 1)} 个平坡区。")

    return resolved_directions


def _collect_unresolved_component(
    start_row: int,
    start_column: int,
    unresolved_mask: np.ndarray,
    visited: np.ndarray,
    neighbor_rows: np.ndarray,
    neighbor_columns: np.ndarray,
) -> list[tuple[int, int]]:
    """Collect one connected unresolved flow component."""

    component_cells: list[tuple[int, int]] = []
    pending_cells: deque[tuple[int, int]] = deque([(start_row, start_column)])
    visited[start_row, start_column] = True

    while pending_cells:
        row_index, column_index = pending_cells.popleft()
        component_cells.append((row_index, column_index))

        for direction_index in range(8):
            neighbor_row = int(neighbor_rows[direction_index, row_index, column_index])
            neighbor_column = int(neighbor_columns[direction_index, row_index, column_index])
            if neighbor_row < 0 or neighbor_column < 0:
                continue
            if visited[neighbor_row, neighbor_column]:
                continue
            if not unresolved_mask[neighbor_row, neighbor_column]:
                continue

            visited[neighbor_row, neighbor_column] = True
            pending_cells.append((neighbor_row, neighbor_column))

    return component_cells


def _trace_flow_outcome(
    start_row: int,
    start_column: int,
    direction_array: np.ndarray,
    valid_mask: np.ndarray,
    component_mask: np.ndarray,
) -> str:
    """Classify whether one downstream trace safely exits or falls into a cycle."""

    height, width = direction_array.shape
    visited_cells: set[tuple[int, int]] = set()
    row_index = start_row
    column_index = start_column

    while True:
        if row_index < 0 or row_index >= height or column_index < 0 or column_index >= width:
            return "exit"
        if not valid_mask[row_index, column_index]:
            return "exit"
        if component_mask[row_index, column_index]:
            return "component"

        current_cell = (row_index, column_index)
        if current_cell in visited_cells:
            return "cycle"
        visited_cells.add(current_cell)

        direction_index = int(direction_array[row_index, column_index])
        if direction_index < 0:
            return "exit"

        row_delta, column_delta = D8_OFFSETS[direction_index]
        row_index += row_delta
        column_index += column_delta


def _choose_component_exit(
    component_cells: list[tuple[int, int]],
    height_array: np.ndarray,
    valid_mask: np.ndarray,
    component_mask: np.ndarray,
    direction_array: np.ndarray,
    allow_unsafe_external: bool = True,
) -> tuple[tuple[int, int], int] | None:
    """Choose the least-cost escape edge for one unresolved component."""

    height, width = height_array.shape
    best_safe_external_candidate: tuple[float, float, int, int, int] | None = None
    best_external_candidate: tuple[float, float, int, int, int] | None = None
    best_border_candidate: tuple[int, int, int] | None = None

    for row_index, column_index in component_cells:
        current_height = float(height_array[row_index, column_index])
        for direction_index, (row_delta, column_delta) in enumerate(D8_OFFSETS):
            neighbor_row = row_index + row_delta
            neighbor_column = column_index + column_delta

            if neighbor_row < 0 or neighbor_row >= height or neighbor_column < 0 or neighbor_column >= width:
                border_candidate = (row_index, column_index, direction_index)
                if best_border_candidate is None or border_candidate < best_border_candidate:
                    best_border_candidate = border_candidate
                continue

            if not valid_mask[neighbor_row, neighbor_column]:
                border_candidate = (row_index, column_index, direction_index)
                if best_border_candidate is None or border_candidate < best_border_candidate:
                    best_border_candidate = border_candidate
                continue
            if component_mask[neighbor_row, neighbor_column]:
                continue

            neighbor_height = float(height_array[neighbor_row, neighbor_column])
            candidate_key = (
                neighbor_height - current_height,
                neighbor_height,
                row_index,
                column_index,
                direction_index,
            )
            trace_outcome = _trace_flow_outcome(
                neighbor_row,
                neighbor_column,
                direction_array,
                valid_mask,
                component_mask,
            )
            if trace_outcome == "exit":
                if best_safe_external_candidate is None or candidate_key < best_safe_external_candidate:
                    best_safe_external_candidate = candidate_key
            if best_external_candidate is None or candidate_key < best_external_candidate:
                best_external_candidate = candidate_key

    if best_safe_external_candidate is not None:
        _, _, row_index, column_index, direction_index = best_safe_external_candidate
        return (row_index, column_index), direction_index

    if best_border_candidate is not None:
        row_index, column_index, direction_index = best_border_candidate
        return (row_index, column_index), direction_index

    if allow_unsafe_external and best_external_candidate is not None:
        _, _, row_index, column_index, direction_index = best_external_candidate
        return (row_index, column_index), direction_index

    return None


def _expand_component_toward_exit(
    seed_cells: list[tuple[int, int]],
    height_array: np.ndarray,
    valid_mask: np.ndarray,
    direction_array: np.ndarray,
    heartbeat_callback: ProgressCallback | None = None,
    heartbeat_label: str = "",
    allow_unsafe_external: bool = True,
) -> tuple[list[tuple[int, int]], np.ndarray, tuple[tuple[int, int], int] | None]:
    """
    Grow one repair component along direct upstream backflow edges until an exit is found.

    This keeps expansion linear in the number of newly absorbed cells instead of
    repeatedly rescanning the full component boundary after every growth step.
    """

    component_mask = np.zeros_like(valid_mask, dtype=bool)
    component_cells: list[tuple[int, int]] = []
    pending_cells: deque[tuple[int, int]] = deque()
    height, width = height_array.shape
    next_heartbeat_at = monotonic() + 0.75

    for row_index, column_index in seed_cells:
        if component_mask[row_index, column_index]:
            continue
        component_mask[row_index, column_index] = True
        component_cells.append((row_index, column_index))
        pending_cells.append((row_index, column_index))

    direct_exit_choice = _choose_component_exit(
        component_cells,
        height_array,
        valid_mask,
        component_mask,
        direction_array,
        allow_unsafe_external=False,
    )
    if direct_exit_choice is not None:
        return component_cells, component_mask, direct_exit_choice

    best_safe_external_candidate: tuple[float, float, int, int, int] | None = None
    best_external_candidate: tuple[float, float, int, int, int] | None = None
    best_border_candidate: tuple[int, int, int] | None = None

    while pending_cells and best_safe_external_candidate is None:
        row_index, column_index = pending_cells.popleft()
        current_height = float(height_array[row_index, column_index])
        next_heartbeat_at = _emit_throttled_heartbeat(
            heartbeat_callback,
            next_heartbeat_at,
            heartbeat_label
            or (
                "流向修复组件扩张："
                f"已吸收 {len(component_cells)} 个像素，待处理队列 {len(pending_cells)}。"
            ),
        )

        for direction_index, (row_delta, column_delta) in enumerate(D8_OFFSETS):
            neighbor_row = row_index + row_delta
            neighbor_column = column_index + column_delta

            if neighbor_row < 0 or neighbor_row >= height or neighbor_column < 0 or neighbor_column >= width:
                border_candidate = (row_index, column_index, direction_index)
                if best_border_candidate is None or border_candidate < best_border_candidate:
                    best_border_candidate = border_candidate
                continue

            if not valid_mask[neighbor_row, neighbor_column]:
                border_candidate = (row_index, column_index, direction_index)
                if best_border_candidate is None or border_candidate < best_border_candidate:
                    best_border_candidate = border_candidate
                continue
            if component_mask[neighbor_row, neighbor_column]:
                continue

            neighbor_direction = int(direction_array[neighbor_row, neighbor_column])
            if neighbor_direction >= 0:
                target_row = neighbor_row + D8_OFFSETS[neighbor_direction][0]
                target_column = neighbor_column + D8_OFFSETS[neighbor_direction][1]
                if (
                    0 <= target_row < height
                    and 0 <= target_column < width
                    and component_mask[target_row, target_column]
                ):
                    component_mask[neighbor_row, neighbor_column] = True
                    component_cells.append((neighbor_row, neighbor_column))
                    pending_cells.append((neighbor_row, neighbor_column))
                    continue

            neighbor_height = float(height_array[neighbor_row, neighbor_column])
            candidate_key = (
                neighbor_height - current_height,
                neighbor_height,
                row_index,
                column_index,
                direction_index,
            )
            trace_outcome = _trace_flow_outcome(
                neighbor_row,
                neighbor_column,
                direction_array,
                valid_mask,
                component_mask,
            )
            if trace_outcome == "exit":
                if (
                    best_safe_external_candidate is None
                    or candidate_key < best_safe_external_candidate
                ):
                    best_safe_external_candidate = candidate_key
            if best_external_candidate is None or candidate_key < best_external_candidate:
                best_external_candidate = candidate_key

    exit_choice: tuple[tuple[int, int], int] | None = None
    if best_safe_external_candidate is not None:
        _, _, row_index, column_index, direction_index = best_safe_external_candidate
        exit_choice = ((row_index, column_index), direction_index)
    elif best_border_candidate is not None:
        row_index, column_index, direction_index = best_border_candidate
        exit_choice = ((row_index, column_index), direction_index)
    elif allow_unsafe_external and best_external_candidate is not None:
        _, _, row_index, column_index, direction_index = best_external_candidate
        exit_choice = ((row_index, column_index), direction_index)

    return component_cells, component_mask, exit_choice


def _assign_component_flow_toward_exit(
    direction_array: np.ndarray,
    component_cells: list[tuple[int, int]],
    anchor_cell: tuple[int, int],
    exit_direction: int,
    height_array: np.ndarray,
    component_mask: np.ndarray,
    neighbor_rows: np.ndarray,
    neighbor_columns: np.ndarray,
) -> None:
    """Route one unresolved component toward its chosen escape edge."""

    distance_map: dict[tuple[int, int], int] = {anchor_cell: 0}
    pending_cells: deque[tuple[int, int]] = deque([anchor_cell])

    while pending_cells:
        row_index, column_index = pending_cells.popleft()
        current_distance = distance_map[(row_index, column_index)]

        for direction_index in range(8):
            neighbor_row = int(neighbor_rows[direction_index, row_index, column_index])
            neighbor_column = int(neighbor_columns[direction_index, row_index, column_index])
            neighbor_cell = (neighbor_row, neighbor_column)
            if neighbor_row < 0 or neighbor_column < 0:
                continue
            if not component_mask[neighbor_row, neighbor_column]:
                continue
            if neighbor_cell in distance_map:
                continue

            distance_map[neighbor_cell] = current_distance + 1
            pending_cells.append(neighbor_cell)

    anchor_row, anchor_column = anchor_cell
    direction_array[anchor_row, anchor_column] = np.int8(exit_direction)

    for row_index, column_index in component_cells:
        if (row_index, column_index) == anchor_cell:
            continue

        current_distance = distance_map[(row_index, column_index)]
        best_direction = -1
        best_key: tuple[float, int] | None = None

        for direction_index in range(8):
            neighbor_row = int(neighbor_rows[direction_index, row_index, column_index])
            neighbor_column = int(neighbor_columns[direction_index, row_index, column_index])
            if neighbor_row < 0 or neighbor_column < 0:
                continue
            if not component_mask[neighbor_row, neighbor_column]:
                continue

            neighbor_distance = distance_map.get((neighbor_row, neighbor_column))
            if neighbor_distance is None or neighbor_distance >= current_distance:
                continue

            candidate_key = (
                float(height_array[neighbor_row, neighbor_column]) - float(height_array[row_index, column_index]),
                direction_index,
            )
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_direction = direction_index

        if best_direction >= 0:
            direction_array[row_index, column_index] = np.int8(best_direction)


def _collect_cycle_components(
    direction_array: np.ndarray,
    valid_mask: np.ndarray,
) -> list[list[tuple[int, int]]]:
    """Return all directed flow cycles that remain after indegree stripping."""

    height, width = direction_array.shape
    downstream_row = np.full((height, width), -1, dtype=np.int32)
    downstream_column = np.full((height, width), -1, dtype=np.int32)
    indegree = np.zeros((height, width), dtype=np.int32)
    active_mask = valid_mask & (direction_array >= 0)

    for row_index in range(height):
        for column_index in range(width):
            if not active_mask[row_index, column_index]:
                continue

            direction_index = int(direction_array[row_index, column_index])
            row_delta, column_delta = D8_OFFSETS[direction_index]
            neighbor_row = row_index + row_delta
            neighbor_column = column_index + column_delta
            if neighbor_row < 0 or neighbor_row >= height or neighbor_column < 0 or neighbor_column >= width:
                continue
            if not active_mask[neighbor_row, neighbor_column]:
                continue

            downstream_row[row_index, column_index] = neighbor_row
            downstream_column[row_index, column_index] = neighbor_column
            indegree[neighbor_row, neighbor_column] += 1

    processing_queue: deque[tuple[int, int]] = deque()
    for row_index in range(height):
        for column_index in range(width):
            if active_mask[row_index, column_index] and indegree[row_index, column_index] == 0:
                processing_queue.append((row_index, column_index))

    while processing_queue:
        row_index, column_index = processing_queue.popleft()
        target_row = int(downstream_row[row_index, column_index])
        target_column = int(downstream_column[row_index, column_index])
        if target_row < 0 or target_column < 0:
            continue

        indegree[target_row, target_column] -= 1
        if indegree[target_row, target_column] == 0:
            processing_queue.append((target_row, target_column))

    cycle_mask = active_mask & (indegree > 0)
    if not cycle_mask.any():
        return []

    cycle_components: list[list[tuple[int, int]]] = []
    visited_cells: set[tuple[int, int]] = set()
    cycle_rows, cycle_columns = np.nonzero(cycle_mask)
    for start_row, start_column in zip(cycle_rows.tolist(), cycle_columns.tolist()):
        start_cell = (start_row, start_column)
        if start_cell in visited_cells:
            continue

        traversal_order: dict[tuple[int, int], int] = {}
        traversal_path: list[tuple[int, int]] = []
        current_cell = start_cell
        while current_cell not in traversal_order:
            traversal_order[current_cell] = len(traversal_path)
            traversal_path.append(current_cell)
            visited_cells.add(current_cell)
            current_row, current_column = current_cell
            next_row = int(downstream_row[current_row, current_column])
            next_column = int(downstream_column[current_row, current_column])
            next_cell = (next_row, next_column)
            if next_row < 0 or next_column < 0 or not cycle_mask[next_row, next_column]:
                current_cell = next_cell
                break
            current_cell = next_cell

        if current_cell in traversal_order:
            cycle_components.append(traversal_path[traversal_order[current_cell] :])

    return cycle_components


def _repair_flow_cycles(
    height_array: np.ndarray,
    valid_mask: np.ndarray,
    direction_array: np.ndarray,
    neighbor_rows: np.ndarray,
    neighbor_columns: np.ndarray,
    progress_callback: ProgressCallback | None = None,
    heartbeat_callback: ProgressCallback | None = None,
) -> np.ndarray:
    """Break residual flow cycles by reconnecting one edge of each cycle to a safe exit."""

    repaired_directions = direction_array.copy()

    for repair_pass in range(1, 17):
        cycle_components = _collect_cycle_components(repaired_directions, valid_mask)
        if not cycle_components:
            if progress_callback is not None:
                progress_callback("D8 4/4 去环修复：未检测到流向环路。")
            return repaired_directions

        current_cycle_node_count = sum(len(component) for component in cycle_components)
        if progress_callback is not None:
            progress_callback(
                (
                    "D8 4/4 去环修复："
                    f"第 {repair_pass} 轮检测到 {len(cycle_components)} 个环，"
                    f"共 {current_cycle_node_count} 个像素。"
                )
            )
        if heartbeat_callback is not None:
            heartbeat_callback(
                (
                    "D8 4/4 去环修复："
                    f"开始处理 {len(cycle_components)} 个环，"
                    f"共 {current_cycle_node_count} 个像素。"
                )
            )

        repaired_cycles = 0
        for cycle_index, cycle_cells in enumerate(cycle_components, start=1):
            repair_cells, component_mask, exit_choice = _expand_component_toward_exit(
                cycle_cells,
                height_array,
                valid_mask,
                repaired_directions,
                heartbeat_callback=heartbeat_callback,
                heartbeat_label=(
                    "D8 4/4 去环修复："
                    f"第 {cycle_index}/{len(cycle_components)} 个环正在扩张修复组件，"
                    f"当前环大小 {len(cycle_cells)}。"
                ),
                allow_unsafe_external=True,
            )
            if exit_choice is None:
                continue

            anchor_cell, exit_direction = exit_choice
            _assign_component_flow_toward_exit(
                repaired_directions,
                repair_cells,
                anchor_cell,
                exit_direction,
                height_array,
                component_mask,
                neighbor_rows,
                neighbor_columns,
            )
            repaired_cycles += 1

            if progress_callback is not None:
                progress_callback(
                    (
                        "D8 4/4 去环修复："
                        f"已修复 {cycle_index}/{len(cycle_components)} 个环。"
                    )
                )

        if repaired_cycles == 0:
            break

    return repaired_directions


def _resolve_residual_unassigned_flows(
    height_array: np.ndarray,
    valid_mask: np.ndarray,
    direction_array: np.ndarray,
    neighbor_rows: np.ndarray,
    neighbor_columns: np.ndarray,
    progress_callback: ProgressCallback | None = None,
    heartbeat_callback: ProgressCallback | None = None,
) -> np.ndarray:
    """Force any remaining unresolved cells to drain toward the cheapest nearby exit."""

    resolved_directions = direction_array.copy()
    unresolved_mask = valid_mask & (resolved_directions < 0)
    if not unresolved_mask.any():
        if progress_callback is not None:
            progress_callback("D8 3/4 断流兜底：未检测到无流向像素，跳过。")
        return resolved_directions

    visited = np.zeros_like(unresolved_mask, dtype=bool)
    unresolved_rows, unresolved_columns = np.nonzero(unresolved_mask)
    total_unresolved = int(unresolved_mask.sum())
    resolved_components = 0
    next_heartbeat_at = monotonic() + 0.75

    if progress_callback is not None:
        progress_callback(
            f"D8 3/4 断流兜底：检测到 {total_unresolved} 个无流向像素，开始强制连通修复。"
        )

    for row_index, column_index in zip(unresolved_rows.tolist(), unresolved_columns.tolist()):
        if visited[row_index, column_index]:
            continue

        component_cells = _collect_unresolved_component(
            row_index,
            column_index,
            unresolved_mask,
            visited,
            neighbor_rows,
            neighbor_columns,
        )
        next_heartbeat_at = _emit_throttled_heartbeat(
            heartbeat_callback,
            next_heartbeat_at,
            (
                "D8 3/4 断流兜底："
                f"正在修复大小为 {len(component_cells)} 的无流向区域，"
                f"剩余无流向像素约 {int((resolved_directions < 0).sum())}/{total_unresolved}。"
            ),
        )
        repair_cells, component_mask, exit_choice = _expand_component_toward_exit(
            component_cells,
            height_array,
            valid_mask,
            resolved_directions,
            heartbeat_callback=heartbeat_callback,
            heartbeat_label=(
                "D8 3/4 断流兜底："
                f"正在扩张大小为 {len(component_cells)} 的无流向区域修复组件。"
            ),
            allow_unsafe_external=True,
        )
        if exit_choice is None:
            continue

        anchor_cell, exit_direction = exit_choice
        _assign_component_flow_toward_exit(
            resolved_directions,
            repair_cells,
            anchor_cell,
            exit_direction,
            height_array,
            component_mask,
            neighbor_rows,
            neighbor_columns,
        )
        resolved_components += 1

        if progress_callback is not None:
            progress_callback(
                (
                    "D8 3/4 断流兜底："
                    f"已修复 {resolved_components} 个无流向区域，"
                    f"当前剩余 {int((resolved_directions < 0).sum())}/{total_unresolved} 个像素。"
                )
            )

    return resolved_directions


def compute_d8_flow_directions(
    height_array: np.ndarray,
    valid_mask: np.ndarray | None = None,
    use_rust_kernel: bool = False,
    slope_weight: float = DEFAULT_SLOPE_WEIGHT,
    flat_escape_weight: float = DEFAULT_FLAT_ESCAPE_WEIGHT,
    outlet_proximity_weight: float = DEFAULT_OUTLET_PROXIMITY_WEIGHT,
    continuity_weight: float = DEFAULT_CONTINUITY_WEIGHT,
    flat_outlet_length_weight: float = DEFAULT_FLAT_OUTLET_LENGTH_WEIGHT,
    flat_outlet_distance_weight: float = DEFAULT_FLAT_OUTLET_DISTANCE_WEIGHT,
    strict_progress_callback: ProgressCallback | None = None,
    flat_progress_callback: ProgressCallback | None = None,
    residual_progress_callback: ProgressCallback | None = None,
    strict_heartbeat_callback: ProgressCallback | None = None,
    flat_heartbeat_callback: ProgressCallback | None = None,
    residual_heartbeat_callback: ProgressCallback | None = None,
    cycle_progress_callback: ProgressCallback | None = None,
    cycle_heartbeat_callback: ProgressCallback | None = None,
) -> np.ndarray:
    """Compute one downstream direction for each terrain cell, including flat routing."""

    height, width = height_array.shape
    working_valid_mask = (
        np.ones((height, width), dtype=bool) if valid_mask is None else valid_mask.astype(bool, copy=False)
    )
    neighbor_rows, neighbor_columns = _build_neighbor_lookup(
        height,
        width,
        heartbeat_callback=strict_heartbeat_callback or flat_heartbeat_callback,
    )
    if use_rust_kernel and rust_kernel_available():
        direction_array = compute_strict_d8_rust(
            height_array,
            working_valid_mask,
            progress_callback=strict_progress_callback or strict_heartbeat_callback,
        )
    else:
        direction_array = _compute_strict_d8_pass(
            height_array,
            working_valid_mask,
            neighbor_rows,
            neighbor_columns,
            slope_weight,
            progress_callback=strict_progress_callback,
            heartbeat_callback=strict_heartbeat_callback,
        )
    direction_array = _resolve_flat_regions(
        height_array,
        working_valid_mask,
        direction_array,
        neighbor_rows,
        neighbor_columns,
        slope_weight,
        flat_escape_weight,
        outlet_proximity_weight,
        continuity_weight,
        flat_outlet_length_weight,
        flat_outlet_distance_weight,
        progress_callback=flat_progress_callback,
        heartbeat_callback=flat_heartbeat_callback,
        use_rust_kernel=use_rust_kernel,
    )
    direction_array = _resolve_residual_unassigned_flows(
        height_array,
        working_valid_mask,
        direction_array,
        neighbor_rows,
        neighbor_columns,
        progress_callback=residual_progress_callback,
        heartbeat_callback=residual_heartbeat_callback or flat_heartbeat_callback,
    )
    return _repair_flow_cycles(
        height_array,
        working_valid_mask,
        direction_array,
        neighbor_rows,
        neighbor_columns,
        progress_callback=cycle_progress_callback,
        heartbeat_callback=cycle_heartbeat_callback or residual_heartbeat_callback or flat_heartbeat_callback,
    )


def direction_preview_image(direction_array: np.ndarray) -> Image.Image:
    """Render the direction raster into a color-coded preview image."""

    height, width = direction_array.shape
    preview_array = np.zeros((height, width, 3), dtype=np.uint8)
    valid_mask = direction_array >= 0
    preview_array[~valid_mask] = np.asarray((18, 24, 32), dtype=np.uint8)

    for direction_index in range(8):
        preview_array[direction_array == direction_index] = DIRECTION_PREVIEW_PALETTE[direction_index]

    return Image.fromarray(preview_array, mode="RGB")


def compute_flow_accumulation(
    direction_array: np.ndarray,
    index_progress_callback: ProgressCallback | None = None,
    propagate_progress_callback: ProgressCallback | None = None,
    heartbeat_callback: ProgressCallback | None = None,
) -> np.ndarray:
    """Compute upstream accumulation by propagating totals downstream."""

    height, width = direction_array.shape
    downstream_row = np.full((height, width), -1, dtype=np.int32)
    downstream_column = np.full((height, width), -1, dtype=np.int32)
    indegree = np.zeros((height, width), dtype=np.int32)
    accumulation = np.ones((height, width), dtype=np.float32)
    next_heartbeat_at = monotonic() + 0.75

    for row_index in range(height):
        for column_index in range(width):
            next_heartbeat_at = _emit_throttled_heartbeat(
                heartbeat_callback,
                next_heartbeat_at,
                f"汇流依赖索引：正在扫描第 {row_index + 1}/{height} 行。",
            )
            direction_index = int(direction_array[row_index, column_index])
            if direction_index < 0:
                continue

            row_delta, column_delta = D8_OFFSETS[direction_index]
            neighbor_row = row_index + row_delta
            neighbor_column = column_index + column_delta
            if neighbor_row < 0 or neighbor_row >= height or neighbor_column < 0 or neighbor_column >= width:
                continue

            downstream_row[row_index, column_index] = neighbor_row
            downstream_column[row_index, column_index] = neighbor_column
            indegree[neighbor_row, neighbor_column] += 1

        if index_progress_callback is not None:
            index_progress_callback(f"汇流依赖索引：已完成 {row_index + 1}/{height} 行。")

    processing_queue: deque[tuple[int, int]] = deque()
    next_heartbeat_at = monotonic() + 0.75
    for row_index in range(height):
        for column_index in range(width):
            if indegree[row_index, column_index] == 0:
                processing_queue.append((row_index, column_index))
            next_heartbeat_at = _emit_throttled_heartbeat(
                heartbeat_callback,
                next_heartbeat_at,
                f"汇流传播准备：正在收集零入度起点，第 {row_index + 1}/{height} 行。",
            )

    processed_nodes = 0
    total_nodes = int(height * width)
    next_heartbeat_at = monotonic() + 0.75

    while processing_queue:
        row_index, column_index = processing_queue.popleft()
        processed_nodes += 1
        target_row = int(downstream_row[row_index, column_index])
        target_column = int(downstream_column[row_index, column_index])

        if heartbeat_callback is not None and monotonic() >= next_heartbeat_at:
            heartbeat_callback(
                f"汇流传播：队列仍在推进，已处理 {processed_nodes}/{total_nodes} 个像素节点。"
            )
            next_heartbeat_at = monotonic() + 0.75

        if target_row < 0 or target_column < 0:
            continue

        accumulation[target_row, target_column] += accumulation[row_index, column_index]
        indegree[target_row, target_column] -= 1
        if indegree[target_row, target_column] == 0:
            processing_queue.append((target_row, target_column))

        if propagate_progress_callback is not None and (
            processed_nodes == 1 or processed_nodes == total_nodes or processed_nodes % max(1, total_nodes // 64) == 0
        ):
            propagate_progress_callback(
                f"汇流传播：已处理 {processed_nodes}/{total_nodes} 个像素节点。"
            )

    cyclic_node_count = int((indegree > 0).sum())
    if cyclic_node_count > 0:
        cycle_message = (
            "汇流传播检测到流向环路："
            f"仍有 {cyclic_node_count} 个像素节点未完成拓扑传播。"
        )
        if heartbeat_callback is not None:
            heartbeat_callback(cycle_message)
        raise ValueError(cycle_message)

    return accumulation


def accumulation_preview_image(
    accumulation: np.ndarray,
    normalize: bool,
) -> Image.Image:
    """Build a preview image for the accumulation raster."""

    transformed = np.log1p(accumulation) if normalize else accumulation.copy()
    minimum_value = float(transformed.min())
    maximum_value = float(transformed.max())
    value_range = maximum_value - minimum_value
    if value_range <= 0:
        normalized = np.zeros_like(transformed, dtype=np.uint8)
    else:
        normalized = ((transformed - minimum_value) / value_range * 255.0).clip(0, 255).astype(np.uint8)

    return Image.fromarray(normalized, mode="L")


def _compute_edge_contact_mask(
    valid_mask: np.ndarray,
    progress_callback: ProgressCallback | None = None,
    heartbeat_callback: ProgressCallback | None = None,
) -> np.ndarray:
    """Return valid pixels that touch the image edge or an invalid-mask neighbor."""

    height, width = valid_mask.shape
    padded_valid = np.pad(valid_mask.astype(bool, copy=False), 1, mode="constant", constant_values=False)
    edge_contact_mask = np.zeros_like(valid_mask, dtype=bool)
    next_heartbeat_at = monotonic() + 0.75

    for direction_index, (row_delta, column_delta) in enumerate(D8_OFFSETS):
        neighbor_valid = padded_valid[
            1 + row_delta : 1 + row_delta + height,
            1 + column_delta : 1 + column_delta + width,
        ]
        edge_contact_mask |= ~neighbor_valid
        next_heartbeat_at = _emit_throttled_heartbeat(
            heartbeat_callback,
            next_heartbeat_at,
            f"河道边缘终止分析：正在检查第 {direction_index + 1}/8 个邻域方向。",
        )
        if progress_callback is not None:
            progress_callback(f"河道边缘终止分析：已完成第 {direction_index + 1}/8 个邻域方向。")

    return edge_contact_mask & valid_mask


def _trim_edge_following_channels(
    channel_mask: np.ndarray,
    direction_array: np.ndarray,
    edge_contact_mask: np.ndarray,
    progress_callback: ProgressCallback | None = None,
    heartbeat_callback: ProgressCallback | None = None,
) -> np.ndarray:
    """Keep only the first edge-touching pixel and trim downstream edge walking."""

    height, width = channel_mask.shape
    candidate_mask = channel_mask.astype(bool, copy=False)
    if not candidate_mask.any():
        return channel_mask.astype(np.uint8, copy=True)

    boundary_channel_mask = candidate_mask & edge_contact_mask
    if not boundary_channel_mask.any():
        if progress_callback is not None:
            progress_callback("河道边缘终止分析：未检测到触边河道，跳过裁剪。")
        return channel_mask.astype(np.uint8, copy=True)

    has_upstream_boundary = np.zeros_like(candidate_mask, dtype=bool)
    has_upstream_interior = np.zeros_like(candidate_mask, dtype=bool)
    next_heartbeat_at = monotonic() + 0.75

    for row_index in range(height):
        for column_index in range(width):
            next_heartbeat_at = _emit_throttled_heartbeat(
                heartbeat_callback,
                next_heartbeat_at,
                f"河道边缘终止分析：正在扫描第 {row_index + 1}/{height} 行的流向连接。",
            )
            if not candidate_mask[row_index, column_index]:
                continue

            direction_index = int(direction_array[row_index, column_index])
            if direction_index < 0:
                continue

            row_delta, column_delta = D8_OFFSETS[direction_index]
            neighbor_row = row_index + row_delta
            neighbor_column = column_index + column_delta
            if neighbor_row < 0 or neighbor_row >= height or neighbor_column < 0 or neighbor_column >= width:
                continue
            if not candidate_mask[neighbor_row, neighbor_column]:
                continue

            if boundary_channel_mask[row_index, column_index]:
                has_upstream_boundary[neighbor_row, neighbor_column] = True
            else:
                has_upstream_interior[neighbor_row, neighbor_column] = True

        if progress_callback is not None:
            progress_callback(f"河道边缘终止分析：已完成 {row_index + 1}/{height} 行流向连接扫描。")

    keep_boundary_mask = boundary_channel_mask & (
        has_upstream_interior | ~(has_upstream_boundary | has_upstream_interior)
    )
    trimmed_mask = (candidate_mask & ~boundary_channel_mask) | keep_boundary_mask

    if progress_callback is not None:
        removed_pixels = int(candidate_mask.sum() - trimmed_mask.sum())
        progress_callback(f"河道边缘终止分析：已裁剪 {removed_pixels} 个沿边缘行走的河道像素。")

    return trimmed_mask.astype(np.uint8, copy=False)


def build_channel_mask(
    accumulation: np.ndarray,
    threshold: float,
    valid_mask: np.ndarray | None = None,
    direction_array: np.ndarray | None = None,
    channel_length_threshold: int = 1,
    progress_callback: ProgressCallback | None = None,
    heartbeat_callback: ProgressCallback | None = None,
) -> np.ndarray:
    """Threshold an accumulation raster into a post-processed binary channel mask."""

    height, width = accumulation.shape
    channel_mask = np.zeros((height, width), dtype=np.uint8)
    next_heartbeat_at = monotonic() + 0.75

    for row_index in range(height):
        channel_mask[row_index, :] = (accumulation[row_index, :] >= threshold).astype(np.uint8)
        if valid_mask is not None:
            channel_mask[row_index, :] = np.where(valid_mask[row_index, :], channel_mask[row_index, :], 0)
        next_heartbeat_at = _emit_throttled_heartbeat(
            heartbeat_callback,
            next_heartbeat_at,
            f"河道阈值提取：正在处理第 {row_index + 1}/{height} 行，阈值 {threshold:.2f}。",
        )
        if progress_callback is not None:
            progress_callback(f"河道阈值提取：已完成 {row_index + 1}/{height} 行。")

    if valid_mask is not None and direction_array is not None:
        edge_contact_mask = _compute_edge_contact_mask(
            valid_mask,
            progress_callback=progress_callback,
            heartbeat_callback=heartbeat_callback,
        )
        channel_mask = _trim_edge_following_channels(
            channel_mask,
            direction_array,
            edge_contact_mask,
            progress_callback=progress_callback,
            heartbeat_callback=heartbeat_callback,
        )

    if channel_length_threshold > 1:
        filtered_channel_mask = _filter_small_components(
            channel_mask.astype(bool, copy=False),
            keep_value=True,
            min_component_size=channel_length_threshold,
            progress_callback=progress_callback,
            heartbeat_callback=heartbeat_callback,
            heartbeat_label="河道长度过滤",
        )
        channel_mask = filtered_channel_mask.astype(np.uint8, copy=False)
        if progress_callback is not None:
            progress_callback(
                f"河道长度过滤：已移除长度小于 {channel_length_threshold} 像素的短河道。"
            )

    return channel_mask


def channel_preview_image(channel_mask: np.ndarray) -> Image.Image:
    """Render the binary channel mask as a high-contrast preview image."""

    height, width = channel_mask.shape
    preview_array = np.zeros((height, width, 3), dtype=np.uint8)
    preview_array[:, :] = np.asarray((10, 16, 20), dtype=np.uint8)
    preview_array[channel_mask > 0] = np.asarray((121, 210, 255), dtype=np.uint8)
    return Image.fromarray(preview_array, mode="RGB")
