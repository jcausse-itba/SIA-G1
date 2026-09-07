use ndarray::Array3;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

#[inline]
fn hcl_to_lab(h: f32, c: f32, l: f32) -> (f32, f32, f32) {
    let rad = h.to_radians();
    let a = c * rad.cos();
    let b = c * rad.sin();
    (l, a, b)
}

#[inline]
fn is_inside(x: i32, y: i32, coords: &[(i32, i32)]) -> bool {
    let mut inside = false;
    let n = coords.len();
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = coords[i];
        let (xj, yj) = coords[j];
        if (yi > y) != (yj > y) {
            let x_intersect = (xj - xi) as f32 * (y - yi) as f32 / (yj - yi) as f32 + xi as f32;
            if (x as f32) < x_intersect {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

#[pyfunction]
#[pyo3(signature = (genomes, width, height))]
fn render_individuals_rust<'py>(
    py: Python<'py>,
    genomes: Vec<PyReadonlyArray2<'py, f32>>,
    width: usize,
    height: usize,
) -> PyResult<Vec<Bound<'py, PyArray3<f32>>>> {
    let views: Vec<_> = genomes.iter().map(|g| g.as_array()).collect();

    let canvases: Vec<Array3<f32>> = py.allow_threads(|| {
        views
            .par_iter()
            .map(|genome| {
                let mut canvas = Array3::<f32>::zeros((height, width, 3));
                canvas.slice_mut(ndarray::s![.., .., 0]).fill(100.0);

                let num_genes = genome.shape()[0];
                let row_len = genome.shape()[1];
                let num_verts = (row_len - 4) / 2;

                for g in 0..num_genes {
                    let row = genome.row(g);
                    let h = row[2 * num_verts];
                    let c = row[2 * num_verts + 1];
                    let l = row[2 * num_verts + 2];
                    let alpha = row[2 * num_verts + 3];

                    let (l_val, a_val, b_val) = hcl_to_lab(h, c, l);
                    let one_minus_alpha = 1.0 - alpha;

                    let mut coords = Vec::with_capacity(num_verts);
                    let mut min_x = width as i32 - 1;
                    let mut max_x = 0i32;
                    let mut min_y = height as i32 - 1;
                    let mut max_y = 0i32;

                    for v in 0..num_verts {
                        let x = (row[2 * v] * width as f32) as i32;
                        let y = (row[2 * v + 1] * height as f32) as i32;
                        coords.push((x, y));

                        if x < min_x { min_x = x; }
                        if x > max_x { max_x = x; }
                        if y < min_y { min_y = y; }
                        if y > max_y { max_y = y; }
                    }

                    min_x = min_x.clamp(0, (width - 1) as i32);
                    max_x = max_x.clamp(0, (width - 1) as i32);
                    min_y = min_y.clamp(0, (height - 1) as i32);
                    max_y = max_y.clamp(0, (height - 1) as i32);

                    for y in min_y..=max_y {
                        for x in min_x..=max_x {
                            if is_inside(x, y, &coords) {
                                let ux = x as usize;
                                let uy = y as usize;
                                canvas[[uy, ux, 0]] = l_val * alpha + canvas[[uy, ux, 0]] * one_minus_alpha;
                                canvas[[uy, ux, 1]] = a_val * alpha + canvas[[uy, ux, 1]] * one_minus_alpha;
                                canvas[[uy, ux, 2]] = b_val * alpha + canvas[[uy, ux, 2]] * one_minus_alpha;
                            }
                        }
                    }
                }
                canvas
            })
            .collect()
    });

    Ok(canvases.into_iter().map(|c| c.into_pyarray(py)).collect())
}

#[pymodule]
fn _tp_2_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(render_individuals_rust, m)?)?;
    Ok(())
}