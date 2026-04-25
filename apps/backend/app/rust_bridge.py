"""
Thin bridge layer for optional Rust acceleration.

The Python pipeline keeps ownership of orchestration, progress reporting, and
fallback behavior. This module only exposes safe wrappers around the compiled
Rust extension so the rest of the backend does not need to know import details.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

try:
    import river_kernel  # type: ignore[import-not-found]
except ImportError:
    river_kernel = None


def rust_kernel_available() -> bool:
    """Return whether the compiled Rust extension is importable."""

    return river_kernel is not None


def rust_priority_flood_available() -> bool:
    """Return whether the installed Rust extension exposes Priority-Flood."""

    return river_kernel is not None and hasattr(river_kernel, "fill_depressions_priority_flood")


def rust_flow_accumulation_available() -> bool:
    """Return whether the installed Rust extension exposes flow accumulation."""

    return river_kernel is not None and hasattr(river_kernel, "compute_flow_accumulation")


def rust_strict_d8_available() -> bool:
    """Return whether the installed Rust extension exposes strict D8 routing."""

    return river_kernel is not None and hasattr(river_kernel, "compute_strict_d8")


def compute_strict_d8_rust(
    height_array: np.ndarray,
    valid_mask: np.ndarray,
    progress_callback: Callable[[str], None] | None = None,
) -> np.ndarray:
    """
    Execute the strict D8 kernel in Rust.

    The caller is expected to handle fallback behavior if the extension is not
    available in the current environment.
    """

    if river_kernel is None:
        raise RuntimeError("Rust kernel is not available in the current Python environment.")

    return np.asarray(
        river_kernel.compute_strict_d8(
            np.asarray(height_array, dtype=np.float32),
            np.asarray(valid_mask, dtype=bool),
            progress_callback,
        ),
        dtype=np.int8,
    )


def label_equal_height_regions_rust(
    height_array: np.ndarray,
    valid_mask: np.ndarray,
    progress_callback: Callable[[str], None] | None = None,
) -> np.ndarray:
    """
    Label 8-connected equal-height regions in Rust.

    The result uses `-1` for invalid cells and non-negative region ids for
    valid equal-height connected components.
    """

    if river_kernel is None:
        raise RuntimeError("Rust kernel is not available in the current Python environment.")

    return np.asarray(
        river_kernel.label_equal_height_regions(
            np.asarray(height_array, dtype=np.float32),
            np.asarray(valid_mask, dtype=bool),
            progress_callback,
        ),
        dtype=np.int32,
    )


def label_connected_components_rust(
    mask_array: np.ndarray,
    progress_callback: Callable[[str], None] | None = None,
) -> np.ndarray:
    """
    Label 8-connected true cells in a boolean mask.

    The result uses `-1` for false cells and non-negative region ids for true
    connected components.
    """

    if river_kernel is None or not hasattr(river_kernel, "label_connected_components"):
        raise RuntimeError("Rust connected-component kernel is not available in the current Python environment.")

    return np.asarray(
        river_kernel.label_connected_components(
            np.asarray(mask_array, dtype=bool),
            progress_callback,
        ),
        dtype=np.int32,
    )


def compute_flat_outlet_drop_map_rust(
    height_array: np.ndarray,
    valid_mask: np.ndarray,
    region_labels: np.ndarray,
    progress_callback: Callable[[str], None] | None = None,
) -> np.ndarray:
    """
    Compute the strongest downhill outlet drop for each labeled flat-region cell.

    Cells without a lower external neighbor receive `0.0`.
    """

    if river_kernel is None:
        raise RuntimeError("Rust kernel is not available in the current Python environment.")

    return np.asarray(
        river_kernel.compute_flat_outlet_drop_map(
            np.asarray(height_array, dtype=np.float32),
            np.asarray(valid_mask, dtype=bool),
            np.asarray(region_labels, dtype=np.int32),
            progress_callback,
        ),
        dtype=np.float32,
    )


def fill_depressions_priority_flood_rust(
    height_array: np.ndarray,
    valid_mask: np.ndarray,
    max_fill_depth: float | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Fill closed depressions using the Rust Priority-Flood kernel."""

    if not rust_priority_flood_available():
        raise RuntimeError("Rust Priority-Flood kernel is not available in the current Python environment.")

    return np.asarray(
        river_kernel.fill_depressions_priority_flood(
            np.asarray(height_array, dtype=np.float32),
            np.asarray(valid_mask, dtype=bool),
            max_fill_depth,
            progress_callback,
        ),
        dtype=np.float32,
    )


def compute_flow_accumulation_rust(
    direction_array: np.ndarray,
    progress_callback: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Compute flow accumulation using the Rust topological propagation kernel."""

    if not rust_flow_accumulation_available():
        raise RuntimeError("Rust flow-accumulation kernel is not available in the current Python environment.")

    return np.asarray(
        river_kernel.compute_flow_accumulation(
            np.asarray(direction_array, dtype=np.int8),
            progress_callback,
        ),
        dtype=np.float32,
    )
