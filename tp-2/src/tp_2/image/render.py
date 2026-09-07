import numpy as np
from PIL import Image, ImageDraw
from tp_2.ga.individual import Individual
from tp_2.ga.color import hcl_to_rgba_int

def render_individual(individual: Individual, width: int, height: int) -> np.ndarray:
    """
    Renderiza un individuo sobre un canvas blanco y devuelve la imagen como array RGB [0, 255].
    """
    canvas = Image.new('RGBA', (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas, 'RGBA')

    for tri in individual.triangles:
        scaled_vertices = [
            (int(x * width), int(y * height)) 
            for x, y in tri.vertices
        ]
        
        # tri.color contiene (H, C, L, Alpha)
        rgba = hcl_to_rgba_int(*tri.color)
        draw.polygon(scaled_vertices, fill=rgba)

    return np.array(canvas.convert('RGB'))