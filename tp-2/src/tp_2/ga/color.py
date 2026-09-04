import numpy as np
import hsluv

def hsluv_to_rgba_int(h: float, s: float, l: float, a: float) -> tuple[int, int, int, int]:
    """
    Convierte el genotipo de color (HSLuv) a formato válido para renderizado de Pillow (RGBA int).
    H: [0, 360], S: [0, 100], L: [0, 100], A: [0, 1]
    """
    # hsluv devuelve r, g, b en rango [0, 1]
    r, g, b = hsluv.hsluv_to_rgb([h, s, l])
    return (int(r * 255), int(g * 255), int(b * 255), int(a * 255))

def rgb_to_lab_vectorized(rgb_img: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen NumPy RGB (H, W, 3) en rango [0, 255] a espacio CIELAB.
    Vectorizado para máximo rendimiento en la evaluación de fitness de la población.
    """
    # 1. Normalizar a [0, 1]
    rgb = rgb_img.astype(np.float32) / 255.0

    # 2. Convertir sRGB a Linear RGB
    mask = rgb > 0.04045
    rgb[mask] = np.power((rgb[mask] + 0.055) / 1.055, 2.4)
    rgb[~mask] = rgb[~mask] / 12.92

    # 3. Linear RGB a XYZ (Matriz para Iluminante D65)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    xyz = np.dot(rgb, M.T)

    # Normalizar por el punto blanco de referencia D65
    xyz_ref_white = np.array([0.95047, 1.00000, 1.08883])
    xyz_normalized = xyz / xyz_ref_white

    # 4. XYZ a CIELAB
    mask2 = xyz_normalized > 0.008856
    f_xyz = np.zeros_like(xyz_normalized)
    f_xyz[mask2] = np.cbrt(xyz_normalized[mask2])
    f_xyz[~mask2] = (7.787 * xyz_normalized[~mask2]) + (16.0 / 116.0)

    lab = np.zeros_like(xyz)
    lab[..., 0] = (116.0 * f_xyz[..., 1]) - 16.0                         # L
    lab[..., 1] = 500.0 * (f_xyz[..., 0] - f_xyz[..., 1])                # a
    lab[..., 2] = 200.0 * (f_xyz[..., 1] - f_xyz[..., 2])                # b

    return lab