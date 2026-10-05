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

    let (m, k) = (x_view.nrows(), x_view.ncols());
    let (k_w, n) = (w_view.nrows(), w_view.ncols());

    assert_eq!(
        k, k_w,
        "Inner dimensions must match for matrix multiplication"
    );

    // Uninitialized output matrix buffer
    let mut result = Array2::<f64>::zeros((m, n));

    // Get raw pointers and strides for matrixmultiply SIMD gemm kernel
    let x_ptr = x_view.as_ptr();
    let w_ptr = w_view.as_ptr();
    let res_ptr = result.as_mut_ptr();

    unsafe {
        matrixmultiply::dgemm(
            m,
            k,
            n,
            1.0, // alpha
            x_ptr,
            x_view.strides()[0],
            x_view.strides()[1], // A strides
            w_ptr,
            w_view.strides()[0],
            w_view.strides()[1], // B strides
            0.0,                 // beta
            res_ptr,
            result.strides()[0],
            result.strides()[1], // C strides
        );
    }

    Ok(result.into_pyarray(py))
}

#[pymodule]
fn rust_matmul(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    Ok(())
}
