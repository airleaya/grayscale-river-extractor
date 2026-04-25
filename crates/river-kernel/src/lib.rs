use pyo3::prelude::*;

mod accumulation;
mod d8;
mod depression;
mod flat;

#[pymodule]
fn river_kernel(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(accumulation::compute_flow_accumulation, module)?)?;
    module.add_function(wrap_pyfunction!(d8::compute_strict_d8, module)?)?;
    module.add_function(wrap_pyfunction!(depression::fill_depressions_priority_flood, module)?)?;
    module.add_function(wrap_pyfunction!(flat::label_connected_components, module)?)?;
    module.add_function(wrap_pyfunction!(flat::label_equal_height_regions, module)?)?;
    module.add_function(wrap_pyfunction!(flat::compute_flat_outlet_drop_map, module)?)?;
    Ok(())
}
