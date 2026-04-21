use pyo3::prelude::*;

mod d8;
mod flat;

#[pymodule]
fn river_kernel(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(d8::compute_strict_d8, module)?)?;
    module.add_function(wrap_pyfunction!(flat::label_connected_components, module)?)?;
    module.add_function(wrap_pyfunction!(flat::label_equal_height_regions, module)?)?;
    module.add_function(wrap_pyfunction!(flat::compute_flat_outlet_drop_map, module)?)?;
    Ok(())
}
