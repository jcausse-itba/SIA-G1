use cust::prelude::*;
use ndarray::Array3;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::Once;

static FIRST_CALL: Once = Once::new();

const CUDA_KERNEL_SRC: &str = r#"
extern "C" __global__ void render_canvas_kernel(
    const float* __restrict__ genomes,
    float* __restrict__ canvases,
    int num_individuals,
    int num_genes,
    int row_len,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int ind = blockIdx.z;

    if (x >= width || y >= height || ind >= num_individuals) return;

    float px = (float)x + 0.5f;
    float py = (float)y + 0.5f;

    int num_verts = (row_len - 4) / 2;

    float cur_l = 100.0f;
    float cur_a = 0.0f;
    float cur_b = 0.0f;

    const float* genome = genomes + (size_t)ind * num_genes * row_len;

    for (int g = 0; g < num_genes; ++g) {
        const float* row = genome + g * row_len;
        float alpha = row[2 * num_verts + 3];

        if (alpha <= 0.0f) continue;

        float min_x = (float)width;
        float max_x = 0.0f;
        float min_y = (float)height;
        float max_y = 0.0f;

        for (int v = 0; v < num_verts; ++v) {
            float vx = row[2 * v] * (float)width;
            float vy = row[2 * v + 1] * (float)height;
            if (vx < min_x) min_x = vx;
            if (vx > max_x) max_x = vx;
            if (vy < min_y) min_y = vy;
            if (vy > max_y) max_y = vy;
        }

        if (px < min_x || px > max_x || py < min_y || py > max_y) {
            continue;
        }

        bool inside = false;
        int j = num_verts - 1;
        for (int i = 0; i < num_verts; ++i) {
            float x1 = row[2 * j] * (float)width;
            float y1 = row[2 * j + 1] * (float)height;
            float x2 = row[2 * i] * (float)width;
            float y2 = row[2 * i + 1] * (float)height;

            if ((y1 <= py && y2 > py) || (y2 <= py && y1 > py)) {
                float x_cross = x1 + (py - y1) / (y2 - y1) * (x2 - x1);
                if (px < x_cross) {
                    inside = !inside;
                }
            }
            j = i;
        }

        if (inside) {
            float h = row[2 * num_verts];
            float c = row[2 * num_verts + 1];
            float l = row[2 * num_verts + 2];

            float rad = h * 0.017453292519943295f;
            float src_l = l * alpha;
            float src_a = (c * cosf(rad)) * alpha;
            float src_b = (c * sinf(rad)) * alpha;
            float one_minus_alpha = 1.0f - alpha;

            cur_l = src_l + cur_l * one_minus_alpha;
            cur_a = src_a + cur_a * one_minus_alpha;
            cur_b = src_b + cur_b * one_minus_alpha;
        }
    }

    size_t out_idx = ((size_t)ind * height * width + (size_t)y * width + (size_t)x) * 3;
    canvases[out_idx] = cur_l;
    canvases[out_idx + 1] = cur_a;
    canvases[out_idx + 2] = cur_b;
}
"#;

fn try_render_gpu(
    py: Python,
    genomes: &[PyReadonlyArray2<f32>],
    width: usize,
    height: usize,
) -> Option<Vec<Array3<f32>>> {
    let _ctx = cust::quick_init().ok()?;

    let num_individuals = genomes.len();
    if num_individuals == 0 {
        return Some(Vec::new());
    }

    let first_genome = genomes[0].as_array();
    let shape = first_genome.shape();
    let num_genes = shape[0];
    let row_len = shape[1];

    let mut flat_genomes = Vec::with_capacity(num_individuals * num_genes * row_len);
    for g in genomes {
        let arr = g.as_array();
        if arr.shape()[0] != num_genes || arr.shape()[1] != row_len {
            return None;
        }
        if let Some(slice) = arr.as_slice() {
            flat_genomes.extend_from_slice(slice);
        } else {
            flat_genomes.extend(arr.iter().copied());
        }
    }

    let module = Module::from_ptx(CUDA_KERNEL_SRC, &[]).ok()?;    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).ok()?;
    let kernel = module.get_function("render_canvas_kernel").ok()?;

    let d_genomes = DeviceBuffer::from_slice(&flat_genomes).ok()?;
    let out_len = num_individuals * height * width * 3;
    let mut d_canvases = unsafe { DeviceBuffer::<f32>::uninitialized(out_len) }.ok()?;

    let block_x = 16u32;
    let block_y = 16u32;
    let grid_x = (width as u32 + block_x - 1) / block_x;
    let grid_y = (height as u32 + block_y - 1) / block_y;
    let grid_z = num_individuals as u32;

    let grid = (grid_x, grid_y, grid_z);
    let block = (block_x, block_y, 1u32);

    py.allow_threads(|| unsafe {
        launch!(
            kernel<<<grid, block, 0, stream>>>(
                d_genomes.as_device_ptr(),
                d_canvases.as_device_ptr(),
                num_individuals as i32,
                num_genes as i32,
                row_len as i32,
                width as i32,
                height as i32
            )
        )
    })
    .ok()?;

    stream.synchronize().ok()?;

    let mut host_canvases = vec![0.0f32; out_len];
    d_canvases.copy_to(&mut host_canvases).ok()?;

    let canvas_size = height * width * 3;
    let mut results = Vec::with_capacity(num_individuals);
    for i in 0..num_individuals {
        let start = i * canvas_size;
        let end = start + canvas_size;
        let slice = host_canvases[start..end].to_vec();
        let arr = Array3::from_shape_vec((height, width, 3), slice).ok()?;
        results.push(arr);
    }

    Some(results)
}

#[inline(always)]
fn hcl_to_lab(h: f32, c: f32, l: f32) -> (f32, f32, f32) {
    let rad = h.to_radians();
    (l, c * rad.cos(), c * rad.sin())
}

#[pyfunction]
#[pyo3(signature = (genomes, width, height))]
fn render_individuals_rust<'py>(
    py: Python<'py>,
    genomes: Vec<PyReadonlyArray2<'py, f32>>,
    width: usize,
    height: usize,
) -> PyResult<Vec<Bound<'py, PyArray3<f32>>>> {
    let population_count = genomes.len();
    let poly_count = if population_count > 0 {
        genomes[0].as_array().shape()[0]
    } else {
        0
    };

    let force_cpu = width <= 128;// && height <= 128 && (poly_count * population_count < 10000);

    if !force_cpu {
        if let Some(canvases) = try_render_gpu(py, &genomes, width, height) {
            FIRST_CALL.call_once(|| {
                println!("Running on GPU");
            });
            return Ok(canvases.into_iter().map(|c| c.into_pyarray(py)).collect());
        }
    }

    FIRST_CALL.call_once(|| {
        println!("Running on CPU");
    });

    let views: Vec<_> = genomes.iter().map(|g| g.as_array()).collect();

    let canvases: Vec<Array3<f32>> = py.allow_threads(|| {
        views
            .par_iter()
            .map(|genome| {
                // Flat allocation avoids multi-index overhead during rasterization
                let mut canvas = vec![0.0f32; height * width * 3];
                
                // Initialize Lightness channel (L*) to 100.0
                for i in (0..canvas.len()).step_by(3) {
                    canvas[i] = 100.0;
                }

                let num_genes = genome.shape()[0];
                let row_len = genome.shape()[1];
                let num_verts = (row_len - 4) / 2;

                // Preallocate reusable buffers per thread
                let mut coords: Vec<(f32, f32)> = Vec::with_capacity(num_verts);
                let mut intersections: Vec<f32> = Vec::with_capacity(num_verts);

                for g in 0..num_genes {
                    let row = genome.row(g);
                    let alpha = row[2 * num_verts + 3];

                    if alpha <= 0.0 {
                        continue;
                    }

                    let h = row[2 * num_verts];
                    let c = row[2 * num_verts + 1];
                    let l = row[2 * num_verts + 2];

                    let (l_val, a_val, b_val) = hcl_to_lab(h, c, l);
                    let src_l = l_val * alpha;
                    let src_a = a_val * alpha;
                    let src_b = b_val * alpha;
                    let one_minus_alpha = 1.0 - alpha;

                    coords.clear();
                    let mut min_y = height as i32;
                    let mut max_y = -1i32;

                    for v in 0..num_verts {
                        let x = row[2 * v] * width as f32;
                        let y = row[2 * v + 1] * height as f32;
                        coords.push((x, y));

                        let y_i = y as i32;
                        if y_i < min_y { min_y = y_i; }
                        if y_i > max_y { max_y = y_i; }
                    }

                    min_y = min_y.clamp(0, height as i32 - 1);
                    max_y = max_y.clamp(0, height as i32 - 1);

                    // Scanline rasterization: loop only over rows covered by the shape
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

                        intersections.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                        // Process filled spans sequentially in memory
                        for chunk in intersections.chunks_exact(2) {
                            let x_start = (chunk[0].ceil() as i32).clamp(0, width as i32 - 1) as usize;
                            let x_end = (chunk[1].floor() as i32).clamp(0, width as i32 - 1) as usize;

                            if x_start <= x_end {
                                let start_idx = (y as usize * width + x_start) * 3;
                                let end_idx = (y as usize * width + x_end + 1) * 3;
                                let span = &mut canvas[start_idx..end_idx];

                                // Contiguous memory access enables compiler SIMD auto-vectorization
                                for pixel in span.chunks_exact_mut(3) {
                                    pixel[0] = src_l + pixel[0] * one_minus_alpha;
                                    pixel[1] = src_a + pixel[1] * one_minus_alpha;
                                    pixel[2] = src_b + pixel[2] * one_minus_alpha;
                                }
                            }
                        }
                    }
                }

                Array3::from_shape_vec((height, width, 3), canvas).unwrap()
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
