import sys
import time
from pathlib import Path

from tp_2.config.parser import build_parser
from tp_2.config.loader import load_and_merge_config
from tp_2.config.validator import validate_config
from tp_2.image.utils import ImageUtils
from tp_2.ga.fitness import FitnessEvaluator
from tp_2.ga.engine import GAEngine
from tp_2.image.render import render_individual

def main() -> None:
    parser = build_parser()

    try:
        cfg = load_and_merge_config(parser)
        validate_config(cfg)
    except Exception as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n[OK] Configuration parsed and validated successfully.")

    # Cargar imagen objetivo
    image_path = getattr(cfg, 'image_path', 'figures/canada-flag.png')
    output_path = getattr(cfg, 'output_path', 'output.png')

    print(f"Cargando imagen: {image_path}")
    target_img = ImageUtils.load_target_image(image_path)

    # Cargar versión liviana para el loop de evolución (128x128 px)
    target_img_eval = ImageUtils.load_target_image(
        image_path, max_size=(128, 128)
    )

    # Cargar también la resolución original solo para guardar el resultado final
    target_img_full = ImageUtils.load_target_image(image_path)

    evaluator = FitnessEvaluator(target_img_eval)

    # Correr motor
    engine = GAEngine(cfg, evaluator)
    start_t = time.time()
    best_individual = engine.run()
    elapsed = time.time() - start_t

    print(f"\nEvolución completada en {elapsed:.2f}s | Mejor Fitness Final: {best_individual.fitness:.4f}")

    # Renderizar y guardar imagen final
    print(f"Guardando imagen generada en {output_path}...")
    rendered_rgb = render_individual(best_individual, evaluator.width, evaluator.height)
    ImageUtils.save_image(rendered_rgb, output_path)
    print("¡Proceso completado exitosamente!")

if __name__ == "__main__":
    main()