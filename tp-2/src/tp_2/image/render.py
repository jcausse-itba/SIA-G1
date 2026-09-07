from typing import List
import numpy as np
import warnings
from numba import njit
from skimage.color import lab2rgb
from tp_2.ga.individual import Individual
from tp_2.ga.color import hcl_to_lab

try:
    from tp_2._tp_2_rust import render_individuals_rust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


@njit(fastmath=True)
def draw_triangle_lab_numba(
    canvas_lab: np.ndarray,
    x0: int, y0: int,
    x1: int, y1: int,
    x2: int, y2: int,
    l_val: float, a_val: float, b_val: float,
    alpha: float
):
    """Rasterizes a triangle using Alpha Blending directly on a CIELAB canvas."""
    height, width, _ = canvas_lab.shape

    min_x = max(0, min(x0, x1, x2))
    max_x = min(width - 1, max(x0, x1, x2))
    min_y = max(0, min(y0, y1, y2))
    max_y = min(height - 1, max(y0, y1, y2))

    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if denom == 0:
        return

    inv_denom = 1.0 / denom
    one_minus_alpha = 1.0 - alpha

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            w0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) * inv_denom
            w1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) * inv_denom
            w2 = 1.0 - w0 - w1

            if w0 >= 0.0 and w1 >= 0.0 and w2 >= 0.0:
                canvas_lab[y, x, 0] = l_val * alpha + canvas_lab[y, x, 0] * one_minus_alpha
                canvas_lab[y, x, 1] = a_val * alpha + canvas_lab[y, x, 1] * one_minus_alpha
                canvas_lab[y, x, 2] = b_val * alpha + canvas_lab[y, x, 2] * one_minus_alpha


def render_individuals(individuals: List[Individual], width: int, height: int, backend: str = "rust") -> List[np.ndarray]:
    """Renders a list of individuals directly onto CIELAB canvases using Rust or Numba."""
    if backend == "rust" and RUST_AVAILABLE:
        genomes = [ind.genome.astype(np.float32) for ind in individuals]
        return render_individuals_rust(genomes, width, height)

    canvases = []
    for individual in individuals:
        canvas_lab = np.zeros((height, width, 3), dtype=np.float32)
        canvas_lab[:, :, 0] = 100.0

        for i in range(individual.genome.shape[0]):
            gene = individual.genome[i]
            
            x0, y0 = int(gene[0] * width), int(gene[1] * height)
            x1, y1 = int(gene[2] * width), int(gene[3] * height)
            x2, y2 = int(gene[4] * width), int(gene[5] * height)

            h, c, l, alpha = gene[6], gene[7], gene[8], gene[9]

            l_val, a_val, b_val = hcl_to_lab(h, c, l)

            draw_triangle_lab_numba(
                canvas_lab,
                x0, y0, x1, y1, x2, y2,
                float(l_val), float(a_val), float(b_val),
                float(alpha)
            )
        canvases.append(canvas_lab)

    return canvases


def render_to_image(individual: Individual, width: int, height: int, backend: str = "rust") -> np.ndarray:
    """Converts rendered CIELAB canvas to sRGB once at the very end."""
    canvas_lab = render_individuals([individual], width, height, backend=backend)[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        canvas_rgb = lab2rgb(canvas_lab)

    return (np.clip(canvas_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
