use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::VecDeque;

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

#[pyfunction(signature = (direction, progress_callback=None))]
pub fn compute_flow_accumulation<'py>(
    py: Python<'py>,
    direction: PyReadonlyArray2<'py, i8>,
    progress_callback: Option<PyObject>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let direction_view = direction.as_array();
    let rows = direction_view.shape()[0];
    let cols = direction_view.shape()[1];
    let total_cells = rows * cols;
    if total_cells == 0 {
        return Ok(Array2::<f32>::zeros((rows, cols)).into_pyarray_bound(py));
    }

    let mut downstream = vec![usize::MAX; total_cells];
    let mut indegree = vec![0u8; total_cells];
    let mut accumulation = vec![1.0f32; total_cells];

    for row in 0..rows {
        for col in 0..cols {
            let direction_index = direction_view[(row, col)];
            if direction_index < 0 {
                continue;
            }
            if direction_index >= 8 {
                return Err(PyValueError::new_err("D8 direction values must be in [-1, 7]."));
            }

            let (row_delta, col_delta) = D8_OFFSETS[direction_index as usize];
            let target_row = row as isize + row_delta;
            let target_col = col as isize + col_delta;
            if target_row < 0
                || target_row >= rows as isize
                || target_col < 0
                || target_col >= cols as isize
            {
                continue;
            }

            let source_index = row * cols + col;
            let target_index = target_row as usize * cols + target_col as usize;
            downstream[source_index] = target_index;
            indegree[target_index] = indegree[target_index].saturating_add(1);
        }

        if row == 0 || (row + 1) % 512 == 0 || row + 1 == rows {
            emit_progress(
                py,
                &progress_callback,
                format!("Rust flow accumulation: indexed {}/{} rows.", row + 1, rows),
            )?;
        }
    }

    let mut queue = VecDeque::<usize>::new();
    for (index, degree) in indegree.iter().enumerate() {
        if *degree == 0 {
            queue.push_back(index);
        }
    }
    emit_progress(
        py,
        &progress_callback,
        format!(
            "Rust flow accumulation: collected {} zero-indegree start cells.",
            queue.len()
        ),
    )?;

    let mut processed = 0usize;
    while let Some(index) = queue.pop_front() {
        processed += 1;
        let target_index = downstream[index];
        if target_index != usize::MAX {
            accumulation[target_index] += accumulation[index];
            indegree[target_index] = indegree[target_index].saturating_sub(1);
            if indegree[target_index] == 0 {
                queue.push_back(target_index);
            }
        }

        if processed == 1 || processed % PROGRESS_INTERVAL == 0 || processed == total_cells {
            emit_progress(
                py,
                &progress_callback,
                format!(
                    "Rust flow accumulation: propagated {}/{} cells.",
                    processed, total_cells
                ),
            )?;
        }
    }

    if processed < total_cells {
        return Err(PyValueError::new_err(format!(
            "flow direction graph contains cycles or unreachable nodes: processed {}/{} cells",
            processed, total_cells
        )));
    }

    let output = Array2::<f32>::from_shape_vec((rows, cols), accumulation)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(output.into_pyarray_bound(py))
}
