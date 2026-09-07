from typing import Tuple
import numpy as np
from PIL import Image

class ImageUtils:

    @staticmethod
    def load_target_image(path: str, max_size: Tuple[int, int] | None = None) -> np.ndarray:
        """Carga la imagen desde disco, la convierte a RGBA y opcionalmente la redimensiona."""
        img = Image.open(path).convert("RGBA")
        if max_size:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
        return np.array(img)

    @staticmethod
    def save_image(image_array: np.ndarray, output_path: str) -> None:
        """Guarda un array NumPy de imagen en disco asegurando el tipo de dato uint8."""
        # 1. Si los valores están normalizados entre 0.0 y 1.0 (float), los reescalamos a 0-255
        if np.issubdtype(image_array.dtype, np.floating):
            if image_array.max() <= 1.0:
                image_array = (image_array * 255.0)
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)

        # 2. Convertir a imagen Pillow y guardar
        img = Image.fromarray(image_array)
        img.save(output_path)