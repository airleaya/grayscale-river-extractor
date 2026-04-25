use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

const D8_OFFSETS: [(isize, isize); 8] = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
];

const PROGRESS_INTERVAL: usize = 1_000_000;

#[derive(Clone, Copy, Debug)]
struct QueueCell {
    level: f32,
    index: usize,
}

impl PartialEq for QueueCell {
    fn eq(&self, other: &Self) -> bool {
        self.level == other.level && self.index == other.index
    }
}

impl Eq for QueueCell {}

impl PartialOrd for QueueCell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueCell {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .level
            .partial_cmp(&self.level)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.index.cmp(&self.index))
    }
}

fn emit_progress(
    py: Python<'_>,
    progress_callback: &Option<PyObject>,
    message: String,
) -> PyResult<()> {
    if let Some(callback) = progress_callback {
        callback.call1(py, (message,))?;
    }

    Ok(())
}

fn is_boundary_valid_cell(valid: &ndarray::ArrayView2<'_, bool>, row: usize, col: usize) -> bool {
    let rows = valid.shape()[0];
    let cols = valid.shape()[1];
    if row == 0 || col == 0 || row + 1 == rows || col + 1 == cols {
        return true;
    }

    for (row_delta, col_delta) in D8_OFFSETS.iter() {
        let neighbor_row = (row as isize + row_delta) as usize;
        let neighbor_col = (col as isize + col_delta) as usize;
        if !valid[(neighbor_row, neighbor_col)] {
            return true;
        }
    }

    false
}

#[pyfunction(signature = (height, valid_mask, max_fill_depth=None, progress_callback=None))]
pub fn fill_depressions_priority_flood<'py>(
    py: Python<'py>,
    height: PyReadonlyArray2<'py, f32>,
    valid_mask: PyReadonlyArray2<'py, bool>,
    max_fill_depth: Option<f32>,
    progress_callback: Option<PyObject>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let height_view = height.as_array();
    let valid_view = valid_mask.as_array();

    if height_view.shape() != valid_view.shape() {
        return Err(PyValueError::new_err(
            "height_array and valid_mask must have the same shape.",
        ));
    }

    let rows = height_view.shape()[0];
    let cols = height_view.shape()[1];
    let total_cells = rows * cols;
    let mut filled = Array2::<f32>::zeros((rows, cols));
    let mut visited = vec![false; total_cells];
    let mut heap = BinaryHeap::<QueueCell>::new();

    for row in 0..rows {
        for col in 0..cols {
            filled[(row, col)] = height_view[(row, col)];
            if !valid_view[(row, col)] {
                continue;
            }
            if is_boundary_valid_cell(&valid_view, row, col) {
                let index = row * cols + col;
                visited[index] = true;
                heap.push(QueueCell {
                    level: height_view[(row, col)],
                    index,
                });
            }
        }
    }

    emit_progress(
        py,
        &progress_callback,
        format!(
            "Rust Priority-Flood: initialized {} outlet cells for {}x{} raster.",
            heap.len(),
            cols,
            rows
        ),
    )?;

    let mut processed = 0usize;
    while let Some(cell) = heap.pop() {
        processed += 1;
        let row = cell.index / cols;
        let col = cell.index % cols;
        let spill_level = filled[(row, col)];

        for (row_delta, col_delta) in D8_OFFSETS.iter() {
            let neighbor_row = row as isize + row_delta;
            let neighbor_col = col as isize + col_delta;
            if neighbor_row < 0
                || neighbor_row >= rows as isize
                || neighbor_col < 0
                || neighbor_col >= cols as isize
            {
                continue;
            }

            let neighbor_row = neighbor_row as usize;
            let neighbor_col = neighbor_col as usize;
            if !valid_view[(neighbor_row, neighbor_col)] {
                continue;
            }

            let neighbor_index = neighbor_row * cols + neighbor_col;
            if visited[neighbor_index] {
                continue;
            }
            visited[neighbor_index] = true;

            let original_height = height_view[(neighbor_row, neighbor_col)];
            let mut filled_height = original_height.max(spill_level);
            if let Some(max_depth) = max_fill_depth {
                if max_depth >= 0.0 {
                    filled_height = filled_height.min(original_height + max_depth);
                }
            }
            filled[(neighbor_row, neighbor_col)] = filled_height;
            heap.push(QueueCell {
                level: filled_height,
                index: neighbor_index,
            });
        }

        if processed == 1 || processed % PROGRESS_INTERVAL == 0 {
            emit_progress(
                py,
                &progress_callback,
                format!(
                    "Rust Priority-Flood: processed {}/{} queued terrain cells.",
                    processed, total_cells
                ),
            )?;
        }
    }

    emit_progress(
        py,
        &progress_callback,
        format!(
            "Rust Priority-Flood: completed depression filling after {} processed cells.",
            processed
        ),
    )?;

    Ok(filled.into_pyarray_bound(py))
}
