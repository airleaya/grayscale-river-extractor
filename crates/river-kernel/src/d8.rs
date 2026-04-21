use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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

const D8_DISTANCES: [f32; 8] = [
    1.0,
    std::f32::consts::SQRT_2,
    1.0,
    std::f32::consts::SQRT_2,
    1.0,
    std::f32::consts::SQRT_2,
    1.0,
    std::f32::consts::SQRT_2,
];

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

#[pyfunction(signature = (height, valid_mask, progress_callback=None))]
pub fn compute_strict_d8<'py>(
    py: Python<'py>,
    height: PyReadonlyArray2<'py, f32>,
    valid_mask: PyReadonlyArray2<'py, bool>,
    progress_callback: Option<PyObject>,
) -> PyResult<Bound<'py, PyArray2<i8>>> {
    let height_view = height.as_array();
    let valid_view = valid_mask.as_array();

    if height_view.shape() != valid_view.shape() {
        return Err(PyValueError::new_err(
            "height_array 和 valid_mask 的形状不一致。",
        ));
    }

    let rows = height_view.shape()[0];
    let cols = height_view.shape()[1];
    let mut direction_array = Array2::<i8>::from_elem((rows, cols), -1);

    for row_index in 0..rows {
        for column_index in 0..cols {
            if !valid_view[(row_index, column_index)] {
                continue;
            }

            let current_height = height_view[(row_index, column_index)];
            let mut best_direction: i8 = -1;
            let mut best_slope = 0.0f32;

            for (direction_index, (row_delta, column_delta)) in D8_OFFSETS.iter().enumerate() {
                let neighbor_row = row_index as isize + row_delta;
                let neighbor_column = column_index as isize + column_delta;
                if neighbor_row < 0
                    || neighbor_row >= rows as isize
                    || neighbor_column < 0
                    || neighbor_column >= cols as isize
                {
                    continue;
                }

                let neighbor_row = neighbor_row as usize;
                let neighbor_column = neighbor_column as usize;
                if !valid_view[(neighbor_row, neighbor_column)] {
                    continue;
                }

                let height_drop = current_height - height_view[(neighbor_row, neighbor_column)];
                if height_drop <= 0.0 {
                    continue;
                }

                let slope = height_drop / D8_DISTANCES[direction_index];
                if slope > best_slope {
                    best_slope = slope;
                    best_direction = direction_index as i8;
                }
            }

            direction_array[(row_index, column_index)] = best_direction;
        }

        emit_progress(
            py,
            &progress_callback,
            format!("Rust 严格 D8：已完成 {}/{} 行。", row_index + 1, rows),
        )?;
    }

    Ok(direction_array.into_pyarray_bound(py))
}
