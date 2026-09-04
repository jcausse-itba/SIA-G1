import numpy as np
from PIL import Image, ImageDraw
from tp_2.ga.individual import Individual
from tp_2.ga.color import hsluv_to_rgba_int

def render_individual(individual: Individual, width: int, height: int) -> np.ndarray:
    """
    Renderiza un individuo sobre un canvas blanco y devuelve la imagen como array RGB.
    """
    # Crear canvas base de color blanco opaco
    canvas = Image.new('RGBA', (width, height), (255, 255, 255, 255))
    
    # Al pasar 'RGBA' al ImageDraw, Pillow soporta alpha-blending básico
    # al superponer polígonos translúcidos sobre colores existentes.
    draw = ImageDraw.Draw(canvas, 'RGBA')

    for tri in individual.triangles:
        # Escalar coordenadas normalizadas [0.0, 1.0] a la resolución objetivo
        scaled_vertices = [
            (int(x * width), int(y * height)) 
            for x, y in tri.vertices
        ]
        
        # Obtener el color ya convertido
        rgba = hsluv_to_rgba_int(*tri.color)
        
        # Dibujar el triángulo
        draw.polygon(scaled_vertices, fill=rgba)
    
    # Convertir a RGB puro (descartando el canal Alpha final) 
    # y exportar como array de NumPy para el cálculo numérico.
    return np.array(canvas.convert('RGB'))