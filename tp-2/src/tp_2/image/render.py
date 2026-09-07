import numpy as np
import warnings
from numba import njit
from skimage.color import lab2rgb
from tp_2.ga.individual import Individual
from tp_2.ga.color import hcl_to_lab

@njit(fastmath=True)
def draw_triangle_lab_numba(canvas_lab: np.ndarray, x0: int, y0: int, x1: int, y1: int, x2: int, y2: int, tri_lab: np.ndarray):
    """Rasteriza escribiendo valores CIELAB directamente sobre el canvas."""
    height, width, _ = canvas_lab.shape

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
                # Asignación directa en espacio CIELAB
                canvas_lab[y, x, 0] = tri_lab[0]
                canvas_lab[y, x, 1] = tri_lab[1]
                canvas_lab[y, x, 2] = tri_lab[2]


def render_individual(individual: Individual, width: int, height: int) -> np.ndarray:
    # Canvas blanco inicializado directamente en CIELAB (L*=100, a*=0, b*=0)
    canvas_lab = np.zeros((height, width, 3), dtype=np.float32)
    canvas_lab[:, :, 0] = 100.0  # L*

    for tri in individual.triangles:
        x0, y0 = int(tri.vertices[0][0] * width), int(tri.vertices[0][1] * height)
        x1, y1 = int(tri.vertices[1][0] * width), int(tri.vertices[1][1] * height)
        x2, y2 = int(tri.vertices[2][0] * width), int(tri.vertices[2][1] * height)

        h, c, l, _ = tri.color
        tri_lab = hcl_to_lab(h, c, l)

        draw_triangle_lab_numba(canvas_lab, x0, y0, x1, y1, x2, y2, tri_lab)

    return canvas_lab

def render_to_image(individual: Individual, width: int, height: int) -> np.ndarray:
    # 1. Renderizado directo en espacio CIELAB
    rendered_lab = render_individual(individual, width, height)

    # 2. Conversión a sRGB [0.0, 1.0] silenciando los warnings de clipping de gamut
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        rendered_rgb = lab2rgb(rendered_lab)

    # 3. Escalar a uint8 [0, 255] para Pillow
    return (np.clip(rendered_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)