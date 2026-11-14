use numpy::ndarray::{Array2, Axis};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
fn matmul<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    w: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x_view = x.as_array();
    let w_view = w.as_array();

    let (m, k) = (x_view.nrows(), x_view.ncols());
    let (k_w, n) = (w_view.nrows(), w_view.ncols());

    if k != k_w {
        return Err(PyValueError::new_err(format!(
            "Inner dimensions must match: ({m}, {k}) @ ({k_w}, {n})"
        )));
    }

    let mut result = Array2::<f64>::zeros((m, n));
    if m == 0 || n == 0 || k == 0 {
        return Ok(result.into_pyarray_bound(py));
    }

    let num_threads = rayon::current_num_threads();

    py.allow_threads(|| {
        if num_threads <= 1 || m * n * k < 64_000 || m < num_threads {
            ndarray::linalg::general_mat_mul(1.0, &x_view, &w_view, 0.0, &mut result);
        } else {
            // Over-chunk into smaller batches to keep Rayon work-stealing efficient across all CPU cores
            let chunk_size = (m / (num_threads * 4)).max(16);

            let res_chunks = result.axis_chunks_iter_mut(Axis(0), chunk_size).into_par_iter();
            let x_chunks = x_view.axis_chunks_iter(Axis(0), chunk_size).into_par_iter();

            res_chunks.zip(x_chunks).for_each(|(mut res_chunk, x_chunk)| {
                ndarray::linalg::general_mat_mul(1.0, &x_chunk, &w_view, 0.0, &mut res_chunk);
            });
        }
    });

    Ok(result.into_pyarray_bound(py))
}

#[pymodule]
fn rust_matmul(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    Ok(())
}
