"""
Focused unit tests for the raster algorithm core.

These tests intentionally use very small synthetic terrains so we can reason
about the expected routing by hand. They form the first validation baseline for
future weighted-flow tuning work.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.raster_algorithms import (
    _compute_gradient_scores,
    _filter_small_components,
    _label_connected_components_python,
    _scale_component_threshold,
    build_channel_mask,
    compute_d8_flow_directions,
    compute_flow_accumulation,
    fill_depressions_priority_flood,
    fill_local_sinks,
    generate_auto_mask,
)
from app.rust_bridge import (
    compute_flow_accumulation_rust,
    compute_strict_d8_rust,
    label_connected_components_rust,
    rust_flow_accumulation_available,
    rust_kernel_available,
    rust_priority_flood_available,
)


class RasterAlgorithmTests(unittest.TestCase):
    """Small deterministic checks for flow-direction behavior."""

    def test_vectorized_gradient_scores_match_naive_d8_reference(self) -> None:
        """The vectorized gradient score should match the original per-pixel D8 scan."""

        terrain = np.asarray(
            (
                (9.0, 8.0, 7.0, 6.0),
                (8.0, 8.0, 7.0, 5.0),
                (7.0, 6.0, 6.0, 4.0),
                (6.0, 5.0, 4.0, 3.0),
            ),
            dtype=np.float32,
        )
        valid_mask = np.asarray(
            (
                (True, True, True, False),
                (True, True, True, True),
                (True, False, True, True),
                (True, True, True, True),
            ),
        )

        gradient_scores = _compute_gradient_scores(terrain, valid_mask)

        gradient = np.zeros_like(terrain, dtype=np.float32)
        for row_index in range(terrain.shape[0]):
            for column_index in range(terrain.shape[1]):
                if not valid_mask[row_index, column_index]:
                    continue
                maximum_delta = 0.0
                for row_delta, column_delta in (
                    (-1, 0),
                    (-1, 1),
                    (0, 1),
                    (1, 1),
                    (1, 0),
                    (1, -1),
                    (0, -1),
                    (-1, -1),
                ):
                    neighbor_row = row_index + row_delta
                    neighbor_column = column_index + column_delta
                    if neighbor_row < 0 or neighbor_row >= terrain.shape[0]:
                        continue
                    if neighbor_column < 0 or neighbor_column >= terrain.shape[1]:
                        continue
                    if not valid_mask[neighbor_row, neighbor_column]:
                        continue
                    maximum_delta = max(
                        maximum_delta,
                        abs(float(terrain[row_index, column_index]) - float(terrain[neighbor_row, neighbor_column])),
                    )
                gradient[row_index, column_index] = maximum_delta

        scale = float(np.percentile(gradient[valid_mask], 95))
        scale = 1.0 if scale <= 1e-6 else scale
        expected_scores = 1.0 - np.clip(gradient / scale, 0.0, 1.0)
        np.testing.assert_allclose(gradient_scores, expected_scores, atol=1e-6)

    def test_strict_d8_prefers_the_steepest_lower_neighbor(self) -> None:
        """The center cell should choose the best downhill route."""

        terrain = np.asarray(
            (
                (30.0, 30.0, 30.0),
                (30.0, 10.0, 1.0),
                (30.0, 30.0, 3.0),
            ),
            dtype=np.float32,
        )

        direction = compute_d8_flow_directions(terrain)

        self.assertEqual(int(direction[1, 1]), 2)

    def test_flat_region_is_routed_toward_a_real_outlet(self) -> None:
        """A long flat strip should be guided toward its only outlet."""

        terrain = np.asarray(
            (
                (12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0),
                (10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 5.0),
                (12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0),
            ),
            dtype=np.float32,
        )

        direction = compute_d8_flow_directions(terrain)

        self.assertEqual(int(direction[1, 0]), 2)
        self.assertEqual(int(direction[1, 1]), 2)
        self.assertEqual(int(direction[1, 2]), 2)
        self.assertEqual(int(direction[1, 3]), 2)
        self.assertEqual(int(direction[1, 4]), 2)
        self.assertEqual(int(direction[1, 5]), 2)

    def test_residual_no_flow_cells_are_force_routed_even_when_flat_weights_are_zero(self) -> None:
        """The final fallback should remove black dots even without flat-routing weights."""

        terrain = np.asarray(
            (
                (12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0),
                (10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 5.0),
                (12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0),
            ),
            dtype=np.float32,
        )

        direction = compute_d8_flow_directions(
            terrain,
            flat_escape_weight=0.0,
            outlet_proximity_weight=0.0,
            continuity_weight=0.0,
        )

        self.assertTrue(np.all(direction >= 0))
        self.assertEqual(int(direction[1, 5]), 2)

    def test_continuity_weight_changes_flat_routing_when_multiple_moves_are_legal(self) -> None:
        """Continuity should change flat routing when several legal candidates exist."""

        terrain = np.asarray(
            (
                (12.0, 12.0, 12.0, 12.0, 12.0),
                (12.0, 10.0, 10.0, 10.0, 7.0),
                (12.0, 10.0, 10.0, 10.0, 12.0),
                (12.0, 12.0, 12.0, 12.0, 12.0),
            ),
            dtype=np.float32,
        )

        baseline_direction = compute_d8_flow_directions(
            terrain,
            flat_escape_weight=1.0,
            outlet_proximity_weight=0.0,
            continuity_weight=0.0,
        )
        continuity_direction = compute_d8_flow_directions(
            terrain,
            flat_escape_weight=1.0,
            outlet_proximity_weight=0.0,
            continuity_weight=1.0,
        )

        self.assertEqual(int(baseline_direction[2, 1]), 1)
        self.assertEqual(int(continuity_direction[2, 1]), 2)

    def test_closed_flat_basin_is_lifted_and_given_a_micro_gradient(self) -> None:
        """A closed flat basin should be raised to spill and sloped toward one anchor."""

        terrain = np.asarray(
            (
                (12.0, 12.0, 12.0, 12.0, 12.0),
                (12.0, 5.0, 5.0, 5.0, 12.0),
                (12.0, 5.0, 5.0, 5.0, 12.0),
                (12.0, 12.0, 12.0, 12.0, 12.0),
            ),
            dtype=np.float32,
        )
        valid_mask = np.ones_like(terrain, dtype=bool)

        repaired, iterations = fill_local_sinks(terrain, valid_mask, enabled=True, max_iterations=4)

        self.assertGreaterEqual(iterations, 1)
        self.assertTrue(float(repaired[1, 1]) > 5.0)
        self.assertTrue(float(repaired[2, 2]) > float(repaired[1, 1]))
        self.assertTrue(np.all(repaired[1:3, 1:4] >= 12.001 - 1e-6))

    def test_single_pixel_sink_is_filled_with_neighbor_average(self) -> None:
        """One-pixel sinks should be raised to the mean of the eight neighbors."""

        terrain = np.asarray(
            (
                (5.0, 6.0, 7.0),
                (8.0, 1.0, 9.0),
                (10.0, 11.0, 12.0),
            ),
            dtype=np.float32,
        )
        valid_mask = np.ones_like(terrain, dtype=bool)

        repaired, iterations = fill_local_sinks(terrain, valid_mask, enabled=True, max_iterations=1)

        self.assertEqual(iterations, 1)
        self.assertAlmostEqual(float(repaired[1, 1]), 8.5, places=5)

    def test_priority_flood_fills_closed_depression_to_spill_level(self) -> None:
        """The fast fill path should raise an enclosed basin to its spill level."""

        if not rust_priority_flood_available():
            self.skipTest("Rust kernel is not installed in the active environment.")

        terrain = np.asarray(
            (
                (9.0, 9.0, 9.0, 9.0, 9.0),
                (9.0, 5.0, 5.0, 5.0, 9.0),
                (9.0, 5.0, 1.0, 5.0, 9.0),
                (9.0, 5.0, 5.0, 5.0, 9.0),
                (9.0, 9.0, 9.0, 9.0, 9.0),
            ),
            dtype=np.float32,
        )
        valid_mask = np.ones_like(terrain, dtype=bool)

        filled, fill_depth = fill_depressions_priority_flood(terrain, valid_mask)

        self.assertEqual(float(filled[2, 2]), 9.0)
        self.assertEqual(float(fill_depth[2, 2]), 8.0)
        self.assertEqual(float(fill_depth[0, 0]), 0.0)

    def test_channel_extraction_prunes_short_components_by_length_threshold(self) -> None:
        """Short channel fragments should be removed from the final channel mask."""

        accumulation = np.asarray(
            (
                (0.0, 8.0, 8.0, 0.0),
                (0.0, 8.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 9.0),
                (0.0, 0.0, 0.0, 0.0),
            ),
            dtype=np.float32,
        )

        channel_mask = build_channel_mask(
            accumulation,
            threshold=7.0,
            channel_length_threshold=2,
        )

        expected = np.asarray(
            (
                (0, 1, 1, 0),
                (0, 1, 0, 0),
                (0, 0, 0, 0),
                (0, 0, 0, 0),
            ),
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(channel_mask, expected)

    def test_channel_extraction_stops_at_image_edge_instead_of_walking_along_it(self) -> None:
        """Only the first edge-touching channel pixel should remain after edge-stop trimming."""

        accumulation = np.zeros((5, 5), dtype=np.float32)
        accumulation[2, 1] = 10.0
        accumulation[1, 1] = 10.0
        accumulation[0, 1] = 10.0
        accumulation[0, 2] = 10.0
        accumulation[0, 3] = 10.0

        direction = np.full((5, 5), -1, dtype=np.int16)
        direction[2, 1] = 0
        direction[1, 1] = 0
        direction[0, 1] = 2
        direction[0, 2] = 2
        direction[0, 3] = 2

        channel_mask = build_channel_mask(
            accumulation,
            threshold=5.0,
            valid_mask=np.ones((5, 5), dtype=bool),
            direction_array=direction,
        )

        self.assertEqual(int(channel_mask[2, 1]), 1)
        self.assertEqual(int(channel_mask[1, 1]), 1)
        self.assertEqual(int(channel_mask[0, 1]), 1)
        self.assertEqual(int(channel_mask[0, 2]), 0)
        self.assertEqual(int(channel_mask[0, 3]), 0)

    def test_channel_extraction_stops_at_mask_edge_instead_of_following_it(self) -> None:
        """Only the first mask-edge contact should remain when a channel reaches the mask boundary."""

        accumulation = np.zeros((5, 5), dtype=np.float32)
        accumulation[2, 2] = 10.0
        accumulation[2, 1] = 10.0
        accumulation[3, 1] = 10.0
        accumulation[4, 1] = 10.0

        valid_mask = np.ones((5, 5), dtype=bool)
        valid_mask[:, 0] = False

        direction = np.full((5, 5), -1, dtype=np.int16)
        direction[2, 2] = 6
        direction[2, 1] = 4
        direction[3, 1] = 4
        direction[4, 1] = -1

        channel_mask = build_channel_mask(
            accumulation,
            threshold=5.0,
            valid_mask=valid_mask,
            direction_array=direction,
        )

        self.assertEqual(int(channel_mask[2, 2]), 1)
        self.assertEqual(int(channel_mask[2, 1]), 1)
        self.assertEqual(int(channel_mask[3, 1]), 0)
        self.assertEqual(int(channel_mask[4, 1]), 0)

    def test_deep_basin_keeps_no_unresolved_black_dots_after_flow_direction_fallback(self) -> None:
        """Residual sink cells should receive fallback directions instead of staying black."""

        terrain = np.asarray(
            (
                (50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0),
                (50.0, 40.0, 40.0, 40.0, 40.0, 40.0, 50.0),
                (50.0, 40.0, 30.0, 30.0, 30.0, 40.0, 50.0),
                (50.0, 40.0, 30.0, 0.0, 30.0, 40.0, 50.0),
                (50.0, 40.0, 30.0, 30.0, 30.0, 40.0, 50.0),
                (50.0, 40.0, 40.0, 40.0, 40.0, 40.0, 50.0),
                (50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0),
            ),
            dtype=np.float32,
        )
        valid_mask = np.ones_like(terrain, dtype=bool)

        repaired, _ = fill_local_sinks(terrain, valid_mask, enabled=True)
        direction = compute_d8_flow_directions(repaired, valid_mask=valid_mask, use_rust_kernel=False)

        self.assertTrue(np.all(direction >= 0))

    def test_residual_fallback_breaks_two_cell_backflow_cycle(self) -> None:
        """Residual repair should not close a 2-cycle when one neighbor already flows into the sink."""

        terrain = np.asarray(
            (
                (5.0, 4.0, 5.0),
                (4.0, 1.0, 2.0),
                (5.0, 4.0, 5.0),
            ),
            dtype=np.float32,
        )
        valid_mask = np.ones_like(terrain, dtype=bool)

        direction = compute_d8_flow_directions(terrain, valid_mask=valid_mask, use_rust_kernel=False)
        accumulation = compute_flow_accumulation(direction)

        self.assertTrue(np.all(direction >= 0))
        self.assertNotEqual((int(direction[1, 1]), int(direction[1, 2])), (2, 6))
        self.assertGreater(float(accumulation[1, 1]), 1.0)

    def test_flow_accumulation_rejects_cyclic_direction_graph(self) -> None:
        """Accumulation should fail loudly when given a cyclic flow graph."""

        direction = np.asarray(
            (
                (-1, -1, -1),
                (-1, 2, 6),
                (-1, -1, -1),
            ),
            dtype=np.int8,
        )

        with self.assertRaisesRegex(ValueError, "流向环路"):
            compute_flow_accumulation(direction)

    def test_rust_flow_accumulation_matches_python_reference(self) -> None:
        """Rust topological propagation should match the Python accumulation result."""

        if not rust_flow_accumulation_available():
            self.skipTest("Rust kernel is not installed in the active environment.")

        direction = np.asarray(
            (
                (2, 2, 4),
                (1, 2, 4),
                (0, 0, -1),
            ),
            dtype=np.int8,
        )

        python_accumulation = compute_flow_accumulation(direction)
        rust_accumulation = compute_flow_accumulation_rust(direction)

        np.testing.assert_allclose(rust_accumulation, python_accumulation, atol=1e-6)

    def test_auto_mask_component_filter_emits_heartbeat_inside_large_component_scan(self) -> None:
        """Large connected-component scans should keep reporting liveness from the BFS loop."""

        mask = np.ones((4, 4), dtype=bool)
        heartbeat_messages: list[str] = []
        fake_times = iter(float(index) for index in range(64))

        with patch('app.raster_algorithms.monotonic', side_effect=lambda: next(fake_times)):
            _filter_small_components(
                mask,
                keep_value=True,
                min_component_size=128,
                heartbeat_callback=heartbeat_messages.append,
                heartbeat_label='自动遮罩连通域过滤',
            )

        self.assertTrue(
            any('正在扩展第 1 个目标连通域' in message for message in heartbeat_messages),
            heartbeat_messages,
        )

    def test_python_component_labeler_tracks_component_sizes_without_cell_lists(self) -> None:
        """The label-based Python fallback should report stable component ids and sizes."""

        mask = np.asarray(
            (
                (True, True, False, False),
                (False, True, False, True),
                (False, False, False, True),
                (True, False, False, False),
            ),
            dtype=bool,
        )

        labels, sizes = _label_connected_components_python(mask)

        self.assertEqual(labels.shape, mask.shape)
        np.testing.assert_array_equal(sizes, np.asarray((3, 2, 1), dtype=np.int32))
        self.assertEqual(int(labels[0, 0]), int(labels[1, 1]))
        self.assertEqual(int(labels[1, 3]), int(labels[2, 3]))
        self.assertNotEqual(int(labels[0, 0]), int(labels[1, 3]))

    def test_filter_small_components_flips_only_components_below_threshold(self) -> None:
        """Component filtering should use label sizes instead of per-component coordinate lists."""

        mask = np.asarray(
            (
                (True, True, False, False),
                (False, True, False, True),
                (False, False, False, True),
                (True, False, False, False),
            ),
            dtype=bool,
        )

        filtered = _filter_small_components(mask, keep_value=True, min_component_size=3)

        expected = np.asarray(
            (
                (True, True, False, False),
                (False, True, False, False),
                (False, False, False, False),
                (False, False, False, False),
            ),
            dtype=bool,
        )
        np.testing.assert_array_equal(filtered, expected)

    def test_auto_mask_downsampled_analysis_preserves_shape_and_boolean_type(self) -> None:
        """Large inputs should be analyzed on a coarse grid but return a full-resolution mask."""

        terrain = np.full((8, 8), 100.0, dtype=np.float32)
        terrain[2:6, 2:6] = 30.0
        valid_mask = np.ones_like(terrain, dtype=bool)
        heartbeat_messages: list[str] = []

        with patch('app.raster_algorithms._choose_auto_mask_downsample_scale', return_value=2):
            auto_mask = generate_auto_mask(
                terrain,
                valid_mask,
                enabled=True,
                border_sensitivity=1.0,
                texture_sensitivity=1.0,
                min_region_size=64,
                heartbeat_callback=heartbeat_messages.append,
            )

        self.assertEqual(auto_mask.shape, terrain.shape)
        self.assertEqual(auto_mask.dtype, np.bool_)
        self.assertTrue(any("自动遮罩降采样预分析" in message for message in heartbeat_messages))
        self.assertEqual(_scale_component_threshold(64, 2), 16)

    def test_rust_connected_component_labels_match_python_fallback(self) -> None:
        """Rust boolean connected-component labeling should match the Python fallback."""

        if not rust_kernel_available():
            self.skipTest("Rust kernel is not installed in the active environment.")

        mask = np.asarray(
            (
                (True, True, False, False),
                (False, True, False, True),
                (False, False, False, True),
                (True, False, False, False),
            ),
            dtype=bool,
        )

        python_labels, python_sizes = _label_connected_components_python(mask)
        rust_labels = label_connected_components_rust(mask)
        rust_valid = rust_labels[rust_labels >= 0]
        rust_sizes = np.bincount(rust_valid, minlength=int(rust_valid.max()) + 1).astype(np.int32)

        np.testing.assert_array_equal(rust_sizes, python_sizes)
        self.assertEqual(int(rust_labels[0, 0]), int(rust_labels[1, 1]))
        self.assertEqual(int(rust_labels[1, 3]), int(rust_labels[2, 3]))
        self.assertEqual(int(rust_labels[3, 0]), int(python_labels[3, 0]))

    def test_rust_flat_region_labeling_keeps_flat_routing_consistent(self) -> None:
        """Rust flat-region labels should preserve Python flat-routing results."""

        if not rust_kernel_available():
            self.skipTest("Rust kernel is not installed in the active environment.")

        terrain = np.asarray(
            (
                (12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0),
                (10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 5.0),
                (12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0),
            ),
            dtype=np.float32,
        )

        python_direction = compute_d8_flow_directions(terrain, use_rust_kernel=False)
        rust_direction = compute_d8_flow_directions(terrain, use_rust_kernel=True)

        np.testing.assert_array_equal(python_direction, rust_direction)

    def test_rust_strict_d8_reports_progress_once_per_row(self) -> None:
        """The Rust strict-D8 kernel should emit one progress callback per row."""

        if not rust_kernel_available():
            self.skipTest("Rust kernel is not installed in the active environment.")

        terrain = np.asarray(
            (
                (9.0, 8.0, 7.0),
                (8.0, 7.0, 6.0),
                (7.0, 6.0, 5.0),
            ),
            dtype=np.float32,
        )
        valid_mask = np.ones_like(terrain, dtype=bool)
        progress_messages: list[str] = []

        compute_strict_d8_rust(
            terrain,
            valid_mask,
            progress_callback=progress_messages.append,
        )

        self.assertEqual(len(progress_messages), terrain.shape[0])
        self.assertEqual(progress_messages[-1], "Rust 严格 D8：已完成 3/3 行。")


if __name__ == "__main__":
    unittest.main()
