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
        return Ok(result.into_pyarray(py));
    }

    let [x_s0, x_s1] = [x_view.strides()[0], x_view.strides()[1]];
    let [w_s0, w_s1] = [w_view.strides()[0], w_view.strides()[1]];
    let [r_s0, r_s1] = [result.strides()[0], result.strides()[1]];

    let num_threads = rayon::current_num_threads();

    if num_threads <= 1 || m * n * k < 64_000 || m < num_threads {
        py.detach(|| unsafe {
            matrixmultiply::dgemm(
                m, k, n, 1.0, x_view.as_ptr(), x_s0, x_s1, w_view.as_ptr(), w_s0, w_s1, 0.0,
                result.as_mut_ptr(), r_s0, r_s1,
            );
        });
        return Ok(result.into_pyarray(py));
    }

    let chunk_size = m / num_threads;

    py.detach(|| {
        result
            .axis_chunks_iter_mut(Axis(0), chunk_size)
            .enumerate()
            .par_bridge()
            .for_each(|(chunk_idx, mut res_chunk)| {
                let row_start = chunk_idx * chunk_size;
                let chunk_rows = res_chunk.nrows();

                unsafe {
                    // Raw pointers are created inside the closure so it stays Send + Sync
                    let x_ptr = x_view.as_ptr().offset(row_start as isize * x_s0);
                    let w_ptr = w_view.as_ptr();
                    let r_ptr = res_chunk.as_mut_ptr();

                    matrixmultiply::dgemm(
                        chunk_rows, k, n, 1.0, x_ptr, x_s0, x_s1, w_ptr, w_s0, w_s1, 0.0, r_ptr,
                        r_s0, r_s1,
                    );
                }
            });
    });

    Ok(result.into_pyarray(py))
}

#[pymodule]
fn rust_matmul(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    Ok(())
}
