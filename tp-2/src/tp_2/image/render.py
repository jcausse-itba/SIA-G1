import numpy as np
from numba import njit
from PIL import Image, ImageDraw
from tp_2.ga.individual import Individual
from tp_2.ga.color import hcl_to_rgb_float_numba, hcl_to_rgba_int, rgb_to_lab_vectorized

@njit(fastmath=True)
def draw_triangle_rgb_numba(canvas_rgb: np.ndarray, x0: int, y0: int, x1: int, y1: int, x2: int, y2: int, tri_rgb: np.ndarray, alpha: float):
    """Rasteriza y realiza alpha blending en espacio sRGB correcto."""
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
                # Alpha blending exacto en RGB
                canvas_rgb[y, x, 0] = canvas_rgb[y, x, 0] * (1.0 - alpha) + tri_rgb[0] * alpha
                canvas_rgb[y, x, 1] = canvas_rgb[y, x, 1] * (1.0 - alpha) + tri_rgb[1] * alpha
                canvas_rgb[y, x, 2] = canvas_rgb[y, x, 2] * (1.0 - alpha) + tri_rgb[2] * alpha


def render_individual_lab_fast(individual: Individual, width: int, height: int) -> np.ndarray:
    canvas_rgb = np.ones((height, width, 3), dtype=np.float32)

    for tri in individual.triangles:
        x0, y0 = int(tri.vertices[0][0] * width), int(tri.vertices[0][1] * height)
        x1, y1 = int(tri.vertices[1][0] * width), int(tri.vertices[1][1] * height)
        x2, y2 = int(tri.vertices[2][0] * width), int(tri.vertices[2][1] * height)

        h, c, l, alpha = tri.color
        tri_rgb = hcl_to_rgb_float_numba(h, c, l)

        draw_triangle_rgb_numba(canvas_rgb, x0, y0, x1, y1, x2, y2, tri_rgb, float(alpha))

    return rgb_to_lab_vectorized(canvas_rgb * 255.0)


def render_individual(individual: Individual, width: int, height: int) -> np.ndarray:
    """Renderiza en Pillow solo para guardar los frames en disco."""
    canvas = Image.new('RGBA', (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas, 'RGBA')

    for tri in individual.triangles:
        scaled_vertices = [
            (int(x * width), int(y * height)) 
            for x, y in tri.vertices
        ]
        rgba = hcl_to_rgba_int(*tri.color)
        draw.polygon(scaled_vertices, fill=rgba)

    return np.array(canvas.convert('RGB'))