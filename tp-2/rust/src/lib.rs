use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::prelude::*;
use ndarray::Array3;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cell::RefCell;
use std::path::Path;
use std::process::Command;
use std::sync::{Once, OnceLock};

static FIRST_CALL: Once = Once::new();
static PTX_CACHE: OnceLock<Result<String, String>> = OnceLock::new();
static CUST_INIT: Once = Once::new();

thread_local! {
    static THREAD_CTX: RefCell<Option<&'static Context>> = const { RefCell::new(None) };
}

fn ensure_cuda_context() -> bool {
    CUST_INIT.call_once(|| {
        let _ = cust::init(cust::CudaFlags::empty());
    });

    THREAD_CTX.with(|ctx_cell| {
        let mut ctx_opt = ctx_cell.borrow_mut();
        if ctx_opt.is_none() {
            if let Ok(device) = Device::get_device(0) {
                if let Ok(ctx) = Context::new(device) {
                    *ctx_opt = Some(Box::leak(Box::new(ctx)));
                }
            }
        }
        ctx_opt.is_some()
    })
}

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
            min_x = fminf(min_x, vx);
            max_x = fmaxf(max_x, vx);
            min_y = fminf(min_y, vy);
            max_y = fmaxf(max_y, vy);
        }

        if (px < min_x || px > max_x || py < min_y || py > max_y) continue;

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

extern "C" __global__ void render_and_fitness_kernel(
    const float* __restrict__ genomes,
    const float* __restrict__ target,
    float* __restrict__ fitnesses,
    int num_individuals,
    int num_genes,
    int row_len,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int ind = blockIdx.z;

    float delta_e = 0.0f;

    if (x < width && y < height && ind < num_individuals) {
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
                min_x = fminf(min_x, vx);
                max_x = fmaxf(max_x, vx);
                min_y = fminf(min_y, vy);
                max_y = fmaxf(max_y, vy);
            }

            if (px < min_x || px > max_x || py < min_y || py > max_y) continue;

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

        int tgt_idx = (y * width + x) * 3;
        float dl = cur_l - target[tgt_idx];
        float da = cur_a - target[tgt_idx + 1];
        float db = cur_b - target[tgt_idx + 2];

        delta_e = sqrtf(dl*dl + da*da + db*db);
    }

    extern __shared__ float sdata[];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    sdata[tid] = delta_e;
    __syncthreads();

    for (unsigned int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&fitnesses[ind], sdata[0]);
    }
}
"#;

fn compile_cuda_to_ptx(src: &str, arch_flag: &str) -> Result<String, String> {
    let temp_dir = std::env::temp_dir();
    let cu_path = temp_dir.join("render_kernel.cu");
    let ptx_path = temp_dir.join("render_kernel.ptx");

    std::fs::write(&cu_path, src).map_err(|e| format!("Failed to write temp CUDA file: {}", e))?;

    let candidate_nvccs = [
        "nvcc",
        "/usr/local/cuda/bin/nvcc",
        "/opt/cuda/bin/nvcc",
        "/usr/bin/nvcc",
    ];

    let mut nvcc_bin = None;
    for bin in candidate_nvccs {
        if Path::new(bin).exists() || Command::new(bin).arg("--version").output().is_ok() {
            nvcc_bin = Some(bin);
            break;
        }
    }

    let bin = nvcc_bin.ok_or_else(|| {
        "nvcc binary not found in PATH or standard CUDA paths (/usr/local/cuda/bin/nvcc)".to_string()
    })?;

    let output = Command::new(bin)
        .args(&[
            "-ptx",
            "-O3",
            arch_flag,
            cu_path.to_str().unwrap(),
            "-o",
            ptx_path.to_str().unwrap(),
        ])
        .output()
        .map_err(|e| format!("Failed to execute nvcc ({}): {}", bin, e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("nvcc compilation failed:\n{}", stderr));
    }

    std::fs::read_to_string(&ptx_path)
        .map_err(|e| format!("Failed to read compiled PTX file: {}", e))
}

fn get_compatible_ptx(major: u32, minor: u32) -> Result<String, String> {
    let mut candidates = vec![format!("-arch=compute_{}{}", major, minor)];
    for fallback in ["-arch=compute_75", "-arch=compute_70", "-arch=compute_60", "-arch=compute_50"] {
        if !candidates.iter().any(|c| c == fallback) {
            candidates.push(fallback.to_string());
        }
    }

    let mut last_err = String::new();
    for arch in &candidates {
        match compile_cuda_to_ptx(CUDA_KERNEL_SRC, arch) {
            Ok(ptx) => match Module::from_ptx(&ptx, &[]) {
                Ok(_) => return Ok(ptx),
                Err(e) => {
                    last_err = format!("Module::from_ptx failed for {}: {:?}", arch, e);
                }
            },
            Err(e) => {
                last_err = e;
            }
        }
    }

    Err(format!("All PTX compilation/loading attempts failed. Last error: {}", last_err))
}

fn try_render_gpu(
    py: Python,
    genomes: &PyReadonlyArray3<f32>,
    width: usize,
    height: usize,
) -> Option<Vec<Array3<f32>>> {
    if !ensure_cuda_context() {
        return None;
    }
    let device = Device::get_device(0).ok()?;

    let major = device.get_attribute(DeviceAttribute::ComputeCapabilityMajor).unwrap_or(6) as u32;
    let minor = device.get_attribute(DeviceAttribute::ComputeCapabilityMinor).unwrap_or(0) as u32;

    let arr = genomes.as_array();
    let shape = arr.shape();
    let num_individuals = shape[0];
    let num_genes = shape[1];
    let row_len = shape[2];

    if num_individuals == 0 {
        return Some(Vec::new());
    }

    let flat_genomes = if let Some(slice) = arr.as_slice() {
        std::borrow::Cow::Borrowed(slice)
    } else {
        std::borrow::Cow::Owned(arr.iter().copied().collect::<Vec<_>>())
    };

    let ptx_res = PTX_CACHE.get_or_init(|| get_compatible_ptx(major, minor));
    let ptx = ptx_res.as_ref().ok()?;
    let module = Module::from_ptx(ptx, &[]).ok()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).ok()?;
    let kernel = module.get_function("render_canvas_kernel").ok()?;

    let d_genomes = DeviceBuffer::from_slice(&flat_genomes).ok()?;

    let out_len = num_individuals * height * width * 3;
    let d_canvases = unsafe { DeviceBuffer::<f32>::uninitialized(out_len).ok()? };

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
    }).ok()?;

    stream.synchronize().ok()?;

    let mut host_canvases = vec![0.0f32; out_len];
    d_canvases.copy_to(&mut host_canvases).ok()?;

    let canvas_size = height * width * 3;
    let mut results = Vec::with_capacity(num_individuals);
    for i in 0..num_individuals {
        let start = i * canvas_size;
        let end = start + canvas_size;
        let slice = host_canvases[start..end].to_vec();
        results.push(Array3::from_shape_vec((height, width, 3), slice).ok()?);
    }

    Some(results)
}

fn try_evaluate_gpu(
    py: Python,
    genomes: &PyReadonlyArray3<f32>,
    target_lab: &PyReadonlyArray3<f32>,
    width: usize,
    height: usize,
) -> Option<Vec<f32>> {
    if !ensure_cuda_context() {
        return None;
    }
    let device = Device::get_device(0).ok()?;

    let major = device.get_attribute(DeviceAttribute::ComputeCapabilityMajor).unwrap_or(6) as u32;
    let minor = device.get_attribute(DeviceAttribute::ComputeCapabilityMinor).unwrap_or(0) as u32;

    let arr = genomes.as_array();
    let shape = arr.shape();
    let num_individuals = shape[0];
    let num_genes = shape[1];
    let row_len = shape[2];

    if num_individuals == 0 {
        return Some(Vec::new());
    }

    let flat_genomes = if let Some(s) = arr.as_slice() {
        std::borrow::Cow::Borrowed(s)
    } else {
        std::borrow::Cow::Owned(arr.iter().copied().collect::<Vec<_>>())
    };

    let target_arr = target_lab.as_array();
    let flat_target = if let Some(s) = target_arr.as_slice() {
        std::borrow::Cow::Borrowed(s)
    } else {
        std::borrow::Cow::Owned(target_arr.iter().copied().collect::<Vec<_>>())
    };

    let ptx_res = PTX_CACHE.get_or_init(|| get_compatible_ptx(major, minor));
    let ptx = ptx_res.as_ref().ok()?;
    let module = Module::from_ptx(ptx, &[]).ok()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).ok()?;

    let fitness_kernel = module.get_function("render_and_fitness_kernel").ok()?;

    let d_genomes = DeviceBuffer::from_slice(&flat_genomes).ok()?;
    let d_target = DeviceBuffer::from_slice(&flat_target).ok()?;

    let mut d_fitnesses = unsafe { DeviceBuffer::<f32>::uninitialized(num_individuals).ok()? };
    let zeros = vec![0.0f32; num_individuals];
    d_fitnesses.copy_from(&zeros).ok()?;

    let block_x = 16u32;
    let block_y = 16u32;
    let grid_x = (width as u32 + block_x - 1) / block_x;
    let grid_y = (height as u32 + block_y - 1) / block_y;
    let grid_z = num_individuals as u32;

    let render_grid = (grid_x, grid_y, grid_z);
    let render_block = (block_x, block_y, 1u32);
    let shared_mem_size = (block_x * block_y * 4) as u32;

    py.allow_threads(|| unsafe {
        launch!(
            fitness_kernel<<<render_grid, render_block, shared_mem_size, stream>>>(
                d_genomes.as_device_ptr(),
                d_target.as_device_ptr(),
                d_fitnesses.as_device_ptr(),
                num_individuals as i32,
                num_genes as i32,
                row_len as i32,
                width as i32,
                height as i32
            )
        )
    }).ok()?;

    stream.synchronize().ok()?;

    let mut host_fitnesses = vec![0.0f32; num_individuals];
    d_fitnesses.copy_to(&mut host_fitnesses).ok()?;

    let num_pixels = (width * height) as f32;
    for fit in host_fitnesses.iter_mut() {
        let mean_delta_e = *fit / num_pixels;
        *fit = 10000.0 / (1.0 + mean_delta_e);
    }

    Some(host_fitnesses)
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
    genomes: PyReadonlyArray3<'py, f32>,
    width: usize,
    height: usize,
) -> PyResult<Vec<Bound<'py, PyArray3<f32>>>> {
    let force_cpu = false;

    if !force_cpu {
        if let Some(canvases) = try_render_gpu(py, &genomes, width, height) {
            FIRST_CALL.call_once(|| {
                println!("Running Render on GPU");
            });
            return Ok(canvases.into_iter().map(|c| c.into_pyarray(py)).collect());
        }
    }

    FIRST_CALL.call_once(|| {
        println!("Running Render on CPU");
    });

    let arr = genomes.as_array();
    let views: Vec<_> = arr.outer_iter().collect();

    let canvases: Vec<Array3<f32>> = py.allow_threads(|| {
        views
            .par_iter()
            .map(|genome| {
                let mut canvas = vec![0.0f32; height * width * 3];

                for i in (0..canvas.len()).step_by(3) {
                    canvas[i] = 100.0;
                }

                let num_genes = genome.shape()[0];
                let row_len = genome.shape()[1];
                let num_verts = (row_len - 4) / 2;

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

                        for chunk in intersections.chunks_exact(2) {
                            let x_start = (chunk[0].ceil() as i32).clamp(0, width as i32 - 1) as usize;
                            let x_end = (chunk[1].floor() as i32).clamp(0, width as i32 - 1) as usize;

                            if x_start <= x_end {
                                let start_idx = (y as usize * width + x_start) * 3;
                                let end_idx = (y as usize * width + x_end + 1) * 3;
                                let span = &mut canvas[start_idx..end_idx];

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

#[pyfunction]
#[pyo3(signature = (genomes, target_lab, width, height))]
fn evaluate_fitness_rust<'py>(
    py: Python<'py>,
    genomes: PyReadonlyArray3<'py, f32>,
    target_lab: PyReadonlyArray3<'py, f32>,
    width: usize,
    height: usize,
) -> PyResult<Vec<f32>> {
    let force_cpu = false;

    if !force_cpu {
        if let Some(fitnesses) = try_evaluate_gpu(py, &genomes, &target_lab, width, height) {
            FIRST_CALL.call_once(|| {
                println!("Running Evaluation on GPU");
            });
            return Ok(fitnesses);
        }
    }

    FIRST_CALL.call_once(|| {
        println!("Running Evaluation on CPU");
    });

    let arr = genomes.as_array();
    let views: Vec<_> = arr.outer_iter().collect();
    let target = target_lab.as_array();

    let fitnesses: Vec<f32> = py.allow_threads(|| {
        views
            .par_iter()
            .map(|genome| {
                let mut canvas = vec![0.0f32; height * width * 3];

                for i in (0..canvas.len()).step_by(3) {
                    canvas[i] = 100.0;
                }

                let num_genes = genome.shape()[0];
                let row_len = genome.shape()[1];
                let num_verts = (row_len - 4) / 2;

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

                        for chunk in intersections.chunks_exact(2) {
                            let x_start = (chunk[0].ceil() as i32).clamp(0, width as i32 - 1) as usize;
                            let x_end = (chunk[1].floor() as i32).clamp(0, width as i32 - 1) as usize;

                            if x_start <= x_end {
                                let start_idx = (y as usize * width + x_start) * 3;
                                let end_idx = (y as usize * width + x_end + 1) * 3;
                                let span = &mut canvas[start_idx..end_idx];

                                for pixel in span.chunks_exact_mut(3) {
                                    pixel[0] = src_l + pixel[0] * one_minus_alpha;
                                    pixel[1] = src_a + pixel[1] * one_minus_alpha;
                                    pixel[2] = src_b + pixel[2] * one_minus_alpha;
                                }
                            }
                        }
                    }
                }

                let mut mean_delta_e = 0.0f32;
                let mut pixel_idx = 0;
                for y in 0..height {
                    for x in 0..width {
                        let dl = canvas[pixel_idx] - target[[y, x, 0]];
                        let da = canvas[pixel_idx + 1] - target[[y, x, 1]];
                        let db = canvas[pixel_idx + 2] - target[[y, x, 2]];
                        mean_delta_e += (dl*dl + da*da + db*db).sqrt();
                        pixel_idx += 3;
                    }
                }
                mean_delta_e /= (height * width) as f32;
                10000.0 / (1.0 + mean_delta_e)
            })
            .collect()
    });

    Ok(fitnesses)
}

#[pymodule]
fn _tp_2_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(render_individuals_rust, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_fitness_rust, m)?)?;
    Ok(())
}
