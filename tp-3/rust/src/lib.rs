use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
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

    assert_eq!(
        k, k_w,
        "Inner dimensions must match for matrix multiplication"
    );

    let mut result = Array2::<f64>::zeros((m, n));

    let x_stride_0 = x_view.strides()[0];
    let x_stride_1 = x_view.strides()[1];
    let w_stride_0 = w_view.strides()[0];
    let w_stride_1 = w_view.strides()[1];

    let res_stride_0 = result.strides()[0];
    let res_stride_1 = result.strides()[1];

    let w_ptr = w_view.as_ptr();

    // Determine row chunk size per Rayon thread task
    let chunk_size = (m / rayon::current_num_threads()).max(1);

    // Parallelize matrix multiplication along the row axis (m) using Rayon & SIMD256
    py.allow_threads(|| {
        result
            .axis_chunks_iter_mut(ndarray::Axis(0), chunk_size)
            .enumerate()
            .par_bridge()
            .for_each(|(chunk_idx, mut res_chunk)| {
                let row_start = chunk_idx * chunk_size;
                let chunk_rows = res_chunk.nrows();

                unsafe {
                    let x_chunk_ptr = x_view.as_ptr().offset(row_start as isize * x_stride_0);
                    let res_chunk_ptr = res_chunk.as_mut_ptr();

                    matrixmultiply::dgemm(
                        chunk_rows,
                        k,
                        n,
                        1.0,
                        x_chunk_ptr,
                        x_stride_0,
                        x_stride_1,
                        w_ptr,
                        w_stride_0,
                        w_stride_1,
                        0.0,
                        res_chunk_ptr,
                        res_stride_0,
                        res_stride_1,
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
