use ndarray::Array3;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

#[inline(always)]
fn hcl_to_lab(h: f32, c: f32, l: f32) -> (f32, f32, f32) {
    let rad = h.to_radians();
    (l, c * rad.cos(), c * rad.sin())
}

#[inline(always)]
fn blend_pixel(pixel: &mut [f32], src_l: f32, src_a: f32, src_b: f32, one_minus_alpha: f32) {
    pixel[0] = src_l + pixel[0] * one_minus_alpha;
    pixel[1] = src_a + pixel[1] * one_minus_alpha;
    pixel[2] = src_b + pixel[2] * one_minus_alpha;
}

#[inline]
fn render_span(
    canvas: &mut [f32],
    y: usize,
    x_start: usize,
    x_end: usize,
    width: usize,
    src_l: f32,
    src_a: f32,
    src_b: f32,
    one_minus_alpha: f32,
) {
    if x_start <= x_end {
        let start_idx = (y * width + x_start) * 3;
        let end_idx = (y * width + x_end + 1) * 3;
        let span = &mut canvas[start_idx..end_idx];

        for pixel in span.chunks_exact_mut(3) {
            blend_pixel(pixel, src_l, src_a, src_b, one_minus_alpha);
        }
    }
}

fn extract_polygon_bounds(
    row: &ndarray::ArrayView1<f32>,
    num_verts: usize,
    width: usize,
    height: usize,
    coords: &mut Vec<(f32, f32)>,
) -> (i32, i32) {
    coords.clear();
    let mut min_y = height as i32;
    let mut max_y = -1i32;

    for v in 0..num_verts {
        let x = row[2 * v] * width as f32;
        let y = row[2 * v + 1] * height as f32;
        coords.push((x, y));

        let y_i = y as i32;
        if y_i < min_y {
            min_y = y_i;
        }
        if y_i > max_y {
            max_y = y_i;
        }
    }

    (
        min_y.clamp(0, height as i32 - 1),
        max_y.clamp(0, height as i32 - 1),
    )
}

fn draw_gene(
    canvas: &mut [f32],
    row: ndarray::ArrayView1<f32>,
    num_verts: usize,
    width: usize,
    height: usize,
    coords: &mut Vec<(f32, f32)>,
    intersections: &mut Vec<f32>,
) {
    let alpha = row[2 * num_verts + 3];
    if alpha <= 0.0 {
        return;
    }

    let h = row[2 * num_verts];
    let c = row[2 * num_verts + 1];
    let l = row[2 * num_verts + 2];

    let (l_val, a_val, b_val) = hcl_to_lab(h, c, l);
    let src_l = l_val * alpha;
    let src_a = a_val * alpha;
    let src_b = b_val * alpha;
    let one_minus_alpha = 1.0 - alpha;

    let (min_y, max_y) = extract_polygon_bounds(&row, num_verts, width, height, coords);

    for y in min_y..=max_y {
        let y_center = y as f32 + 0.5;
        intersections.clear();

        let mut j = num_verts - 1;
        for i in 0..num_verts {
            let (x1, y1) = coords[j];
            let (x2, y2) = coords[i];

            if (y1 <= y_center && y2 > y_center) || (y2 <= y_center && y1 > y_center) {
                let t = (y_center - y1) / (y2 - y1);
                intersections.push(x1 + t * (x2 - x1));
            }
            j = i;
        }

        intersections
            .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        for chunk in intersections.chunks_exact(2) {
            let x_start = (chunk[0].ceil() as i32).clamp(0, width as i32 - 1) as usize;
            let x_end = (chunk[1].floor() as i32).clamp(0, width as i32 - 1) as usize;

            render_span(
                canvas,
                y as usize,
                x_start,
                x_end,
                width,
                src_l,
                src_a,
                src_b,
                one_minus_alpha,
            );
        }
    }
}

fn render_single_individual(
    genome: ndarray::ArrayView2<f32>,
    width: usize,
    height: usize,
) -> Array3<f32> {
    let mut canvas = vec![0.0f32; height * width * 3];

    // Initialize Lightness channel (L*) to 100.0
    for i in (0..canvas.len()).step_by(3) {
        canvas[i] = 100.0;
    }

    let num_genes = genome.shape()[0];
    let row_len = genome.shape()[1];
    let num_verts = (row_len - 4) / 2;

    let mut coords = Vec::with_capacity(num_verts);
    let mut intersections = Vec::with_capacity(num_verts);

    for g in 0..num_genes {
        draw_gene(
            &mut canvas,
            genome.row(g),
            num_verts,
            width,
            height,
            &mut coords,
            &mut intersections,
        );
    }

    Array3::from_shape_vec((height, width, 3), canvas).unwrap()
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
            .map(|genome| render_single_individual(*genome, width, height))
            .collect()
    });

    Ok(canvases.into_iter().map(|c| c.into_pyarray(py)).collect())
}

#[pymodule]
fn _tp_2_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(render_individuals_rust, m)?)?;
    Ok(())
}
