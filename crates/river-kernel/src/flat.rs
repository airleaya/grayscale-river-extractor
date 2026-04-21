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

const HEIGHT_EPSILON: f32 = 1e-6;
const COMPONENT_HEARTBEAT_INTERVAL: usize = 4096;

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

#[pyfunction(signature = (mask, progress_callback=None))]
pub fn label_connected_components<'py>(
    py: Python<'py>,
    mask: PyReadonlyArray2<'py, bool>,
    progress_callback: Option<PyObject>,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let mask_view = mask.as_array();
    let rows = mask_view.shape()[0];
    let cols = mask_view.shape()[1];
    let mut label_array = Array2::<i32>::from_elem((rows, cols), -1);
    let mut current_label: i32 = 0;

    for row_index in 0..rows {
        for column_index in 0..cols {
            if !mask_view[(row_index, column_index)] {
                continue;
            }
            if label_array[(row_index, column_index)] >= 0 {
                continue;
            }

            let component_number = current_label + 1;
            let mut component_size: usize = 0;
            let mut pending_cells = VecDeque::<(usize, usize)>::new();
            pending_cells.push_back((row_index, column_index));
            label_array[(row_index, column_index)] = current_label;

            while let Some((current_row, current_column)) = pending_cells.pop_front() {
                component_size += 1;
                if component_size == 1 || component_size % COMPONENT_HEARTBEAT_INTERVAL == 0 {
                    emit_progress(
                        py,
                        &progress_callback,
                        format!(
                            "HEARTBEAT|自动遮罩连通域标记：正在扩展第 {} 个目标连通域，已扫描 {} 个像素，待处理队列 {}。",
                            component_number,
                            component_size,
                            pending_cells.len()
                        ),
                    )?;
                }

                for (row_delta, column_delta) in D8_OFFSETS.iter() {
                    let neighbor_row = current_row as isize + row_delta;
                    let neighbor_column = current_column as isize + column_delta;
                    if neighbor_row < 0
                        || neighbor_row >= rows as isize
                        || neighbor_column < 0
                        || neighbor_column >= cols as isize
                    {
                        continue;
                    }

                    let neighbor_row = neighbor_row as usize;
                    let neighbor_column = neighbor_column as usize;
                    if !mask_view[(neighbor_row, neighbor_column)] {
                        continue;
                    }
                    if label_array[(neighbor_row, neighbor_column)] >= 0 {
                        continue;
                    }

                    label_array[(neighbor_row, neighbor_column)] = current_label;
                    pending_cells.push_back((neighbor_row, neighbor_column));
                }
            }

            emit_progress(
                py,
                &progress_callback,
                format!(
                    "HEARTBEAT|自动遮罩连通域标记：第 {} 个目标连通域扫描完成，大小 {} 像素。",
                    component_number,
                    component_size
                ),
            )?;
            current_label += 1;
        }

        emit_progress(
            py,
            &progress_callback,
            format!("ROW|自动遮罩连通域标记：已完成 {}/{} 行。", row_index + 1, rows),
        )?;
    }

    Ok(label_array.into_pyarray_bound(py))
}

#[pyfunction(signature = (height, valid_mask, progress_callback=None))]
pub fn label_equal_height_regions<'py>(
    py: Python<'py>,
    height: PyReadonlyArray2<'py, f32>,
    valid_mask: PyReadonlyArray2<'py, bool>,
    progress_callback: Option<PyObject>,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let height_view = height.as_array();
    let valid_view = valid_mask.as_array();

    if height_view.shape() != valid_view.shape() {
        return Err(PyValueError::new_err(
            "height_array and valid_mask must have the same shape.",
        ));
    }

    let rows = height_view.shape()[0];
    let cols = height_view.shape()[1];
    let mut label_array = Array2::<i32>::from_elem((rows, cols), -1);
    let mut current_label: i32 = 0;

    for row_index in 0..rows {
        for column_index in 0..cols {
            if !valid_view[(row_index, column_index)] {
                continue;
            }
            if label_array[(row_index, column_index)] >= 0 {
                continue;
            }

            let target_height = height_view[(row_index, column_index)];
            let mut pending_cells = VecDeque::<(usize, usize)>::new();
            pending_cells.push_back((row_index, column_index));
            label_array[(row_index, column_index)] = current_label;

            while let Some((current_row, current_column)) = pending_cells.pop_front() {
                for (row_delta, column_delta) in D8_OFFSETS.iter() {
                    let neighbor_row = current_row as isize + row_delta;
                    let neighbor_column = current_column as isize + column_delta;
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
                    if label_array[(neighbor_row, neighbor_column)] >= 0 {
                        continue;
                    }

                    let neighbor_height = height_view[(neighbor_row, neighbor_column)];
                    if (neighbor_height - target_height).abs() > HEIGHT_EPSILON {
                        continue;
                    }

                    label_array[(neighbor_row, neighbor_column)] = current_label;
                    pending_cells.push_back((neighbor_row, neighbor_column));
                }
            }

            current_label += 1;
        }

        emit_progress(
            py,
            &progress_callback,
            format!("Rust 平坡分区标记：已完成 {}/{} 行。", row_index + 1, rows),
        )?;
    }

    Ok(label_array.into_pyarray_bound(py))
}

#[pyfunction(signature = (height, valid_mask, region_labels, progress_callback=None))]
pub fn compute_flat_outlet_drop_map<'py>(
    py: Python<'py>,
    height: PyReadonlyArray2<'py, f32>,
    valid_mask: PyReadonlyArray2<'py, bool>,
    region_labels: PyReadonlyArray2<'py, i32>,
    progress_callback: Option<PyObject>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let height_view = height.as_array();
    let valid_view = valid_mask.as_array();
    let label_view = region_labels.as_array();

    if height_view.shape() != valid_view.shape() || height_view.shape() != label_view.shape() {
        return Err(PyValueError::new_err(
            "height_array, valid_mask, and region_labels must have the same shape.",
        ));
    }

    let rows = height_view.shape()[0];
    let cols = height_view.shape()[1];
    let mut drop_map = Array2::<f32>::zeros((rows, cols));

    for row_index in 0..rows {
        for column_index in 0..cols {
            if !valid_view[(row_index, column_index)] {
                continue;
            }

            let current_label = label_view[(row_index, column_index)];
            if current_label < 0 {
                continue;
            }

            let current_height = height_view[(row_index, column_index)];
            let mut strongest_drop = 0.0f32;

            for (row_delta, column_delta) in D8_OFFSETS.iter() {
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
                if label_view[(neighbor_row, neighbor_column)] == current_label {
                    continue;
                }

                let height_drop = current_height - height_view[(neighbor_row, neighbor_column)];
                if height_drop > strongest_drop {
                    strongest_drop = height_drop;
                }
            }

            drop_map[(row_index, column_index)] = strongest_drop.max(0.0);
        }

        emit_progress(
            py,
            &progress_callback,
            format!("Rust 平坡出口落差图：已完成 {}/{} 行。", row_index + 1, rows),
        )?;
    }

    Ok(drop_map.into_pyarray_bound(py))
}
