import numpy as np
from numba import njit

def hcl_to_lab(h: float, c: float, l: float) -> np.ndarray:
    """Convierte genotipo HCL directamente a un vector CIELAB [L*, a*, b*]."""
    h_rad = np.radians(h)
    a = c * np.cos(h_rad)
    b = c * np.sin(h_rad)
    return np.array([l, a, b], dtype=np.float32)


def hcl_to_rgba_int(h: float, c: float, l: float, alpha: float) -> tuple[int, int, int, int]:
    """Conversión completa HCL a sRGB [0, 255] para renderizado y exportación."""
    h_rad = np.radians(h)
    a = c * np.cos(h_rad)
    b = c * np.sin(h_rad)

    fy = (l + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    fx3, fy3, fz3 = fx**3, fy**3, fz**3
    x_val = fx3 if fx3 > 0.008856 else (fx - 16.0 / 116.0) / 7.787
    y_val = fy3 if fy3 > 0.008856 else (fy - 16.0 / 116.0) / 7.787
    z_val = fz3 if fz3 > 0.008856 else (fz - 16.0 / 116.0) / 7.787

    xyz_ref_white = np.array([0.95047, 1.00000, 1.08883])
    xyz = np.array([x_val, y_val, z_val]) * xyz_ref_white

    M_inv = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ])
    rgb_linear = np.dot(M_inv, xyz)

    mask = rgb_linear > 0.0031308
    rgb = np.zeros_like(rgb_linear)
    rgb[mask] = 1.055 * (rgb_linear[mask] ** (1.0 / 2.4)) - 0.055
    rgb[~mask] = 12.92 * rgb_linear[~mask]

    rgb_clamped = np.clip(rgb, 0.0, 1.0)
    return (
        int(rgb_clamped[0] * 255),
        int(rgb_clamped[1] * 255),
        int(rgb_clamped[2] * 255),
        int(alpha * 255)
    )


@njit(fastmath=True)
def hcl_to_rgb_float_numba(h: float, c: float, l: float) -> np.ndarray:
    """Conversión ultrarrápida HCL -> sRGB [0.0, 1.0] compatible con Numba."""
    h_rad = np.radians(h)
    a = c * np.cos(h_rad)
    b = c * np.sin(h_rad)

    fy = (l + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    fx3, fy3, fz3 = fx**3, fy**3, fz**3
    x_val = fx3 if fx3 > 0.008856 else (fx - 16.0 / 116.0) / 7.787
    y_val = fy3 if fy3 > 0.008856 else (fy - 16.0 / 116.0) / 7.787
    z_val = fz3 if fz3 > 0.008856 else (fz - 16.0 / 116.0) / 7.787

    # XYZ (D65)
    x = x_val * 0.95047
    y = y_val * 1.00000
    z = z_val * 1.08883

    # XYZ a sRGB Lineal
    r_lin =  3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g_lin = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b_lin =  0.0556434 * x - 0.2040259 * y + 1.0572252 * z

    # Compresión Gamma
    r = 1.055 * (r_lin ** (1.0 / 2.4)) - 0.055 if r_lin > 0.0031308 else 12.92 * r_lin
    g = 1.055 * (g_lin ** (1.0 / 2.4)) - 0.055 if g_lin > 0.0031308 else 12.92 * g_lin
    b = 1.055 * (b_lin ** (1.0 / 2.4)) - 0.055 if b_lin > 0.0031308 else 12.92 * b_lin

    return np.array([
        max(0.0, min(1.0, r)),
        max(0.0, min(1.0, g)),
        max(0.0, min(1.0, b))
    ], dtype=np.float32)


def rgb_to_lab_vectorized(rgb_img: np.ndarray) -> np.ndarray:
    """Convierte la imagen objetivo NumPy RGB a CIELAB (se usa una sola vez)."""
    rgb = rgb_img.astype(np.float32) / 255.0

    mask = rgb > 0.04045
    rgb[mask] = np.power((rgb[mask] + 0.055) / 1.055, 2.4)
    rgb[~mask] = rgb[~mask] / 12.92

    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    xyz = np.dot(rgb, M.T)

    xyz_ref_white = np.array([0.95047, 1.00000, 1.08883])
    xyz_normalized = xyz / xyz_ref_white

    mask2 = xyz_normalized > 0.008856
    f_xyz = np.zeros_like(xyz_normalized)
    f_xyz[mask2] = np.cbrt(xyz_normalized[mask2])
    f_xyz[~mask2] = (7.787 * xyz_normalized[~mask2]) + (16.0 / 116.0)

    lab = np.zeros_like(xyz)
    lab[..., 0] = (116.0 * f_xyz[..., 1]) - 16.0
    lab[..., 1] = 500.0 * (f_xyz[..., 0] - f_xyz[..., 1])
    lab[..., 2] = 200.0 * (f_xyz[..., 1] - f_xyz[..., 2])

    return lab.astype(np.float32)