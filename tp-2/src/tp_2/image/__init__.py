from typing import Tuple
import numpy as np
from PIL import Image

# TODO don't like the name...
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
        """Guarda un array NumPy de imagen en disco."""
        img = Image.fromarray(image_array)
        img.save(output_path)