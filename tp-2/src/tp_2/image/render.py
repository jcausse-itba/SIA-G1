import numpy as np
import warnings
from numba import njit
from skimage.color import lab2rgb
from tp_2.ga.individual import Individual
from tp_2.ga.color import hcl_to_lab

@njit(fastmath=True)
def draw_triangle_rgb_numba(
    canvas_rgb: np.ndarray,
    x0: int, y0: int,
    x1: int, y1: int,
    x2: int, y2: int,
    tri_rgb: np.ndarray,
    alpha: float
):
    """Rasteriza aplicando Alpha Blending sobre un canvas RGB [0.0, 1.0]."""
    height, width, _ = canvas_rgb.shape

    min_x = max(0, min(x0, x1, x2))
    max_x = min(width - 1, max(x0, x1, x2))
    min_y = max(0, min(y0, y1, y2))
    max_y = min(height - 1, max(y0, y1, y2))

    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if denom == 0:
        return

    inv_denom = 1.0 / denom

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            w0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) * inv_denom
            w1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) * inv_denom
            w2 = 1.0 - w0 - w1

            if w0 >= 0 and w1 >= 0 and w2 >= 0:
                canvas_rgb[y, x, 0] = tri_rgb[0] * alpha + canvas_rgb[y, x, 0] * (1.0 - alpha)
                canvas_rgb[y, x, 1] = tri_rgb[1] * alpha + canvas_rgb[y, x, 1] * (1.0 - alpha)
                canvas_rgb[y, x, 2] = tri_rgb[2] * alpha + canvas_rgb[y, x, 2] * (1.0 - alpha)


def render_individual(individual: Individual, width: int, height: int) -> np.ndarray:
    canvas_rgb = np.ones((height, width, 3), dtype=np.float32)

    for tri in individual.triangles:
        x0, y0 = int(tri.vertices[0][0] * width), int(tri.vertices[0][1] * height)
        x1, y1 = int(tri.vertices[1][0] * width), int(tri.vertices[1][1] * height)
        x2, y2 = int(tri.vertices[2][0] * width), int(tri.vertices[2][1] * height)

        h, c, l, alpha = tri.color

        tri_lab = hcl_to_lab(h, c, l)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            tri_rgb = lab2rgb(tri_lab.reshape((1, 1, 3))).reshape(3)

        draw_triangle_rgb_numba(canvas_rgb, x0, y0, x1, y1, x2, y2, tri_rgb, alpha)

    from skimage.color import rgb2lab
    return rgb2lab(canvas_rgb).astype(np.float32)

def render_to_image(individual: Individual, width: int, height: int) -> np.ndarray:
    # Si render_individual retorna CIELAB para la evaluación de fitness:
    rendered_lab = render_individual(individual, width, height)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        rendered_rgb = lab2rgb(rendered_lab)

    return (np.clip(rendered_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)