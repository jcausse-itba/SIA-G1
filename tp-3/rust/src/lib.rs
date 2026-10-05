use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyfunction]
fn matmul<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    w: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x_view = x.as_array();
    let w_view = w.as_array();

    // Matrix multiplication: (m, k) @ (k, n) -> (m, n)
    let result: Array2<f64> = x_view.dot(&w_view);

    Ok(result.into_pyarray(py))
}

#[pymodule]
fn rust_matmul(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    Ok(())
}
