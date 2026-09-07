import sys
from tp_2.config.loader import load_and_merge_config
from tp_2.config.parser import build_parser
from tp_2.config.validator import validate_config
from tp_2.ga.engine import GAEngine
from tp_2.ga.fitness import FitnessEvaluator
from tp_2.image.render import render_to_image
from tp_2.image.utils import ImageUtils


def main() -> None:
    parser = build_parser()

    try:
        cfg = load_and_merge_config(parser)
        validate_config(cfg)
    except Exception as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n[OK] Configuration parsed and validated successfully.")

    image_path = cfg.get("image_path", cfg.get("image", None))
    if not image_path:
        image_path = "figures/germany-flag.png"

    output_path = cfg.get("output_path", "output.png")

    print(f"Cargando imagen: {image_path}")

    # 1. Cargar imagen original completa para saber sus dimensiones nativas
    target_img_full = ImageUtils.load_target_image(image_path)
    full_height, full_width = target_img_full.shape[:2]

    # 2. Cargar imagen en baja resolución solo para acelerar la función de fitness
    eval_size = cfg.get("eval_size", 128)
    target_img_eval = ImageUtils.load_target_image(
        image_path, max_size=(eval_size, eval_size)
    )

    evaluator = FitnessEvaluator(target_img_eval)
    engine = GAEngine(cfg, evaluator)

    # 3. Ejecutar algoritmo genético
    best_individual = engine.run()

    # 4. Renderizado final usando las dimensiones completas (full_width, full_height)
    final_render = render_to_image(
        best_individual, full_width, full_height
    )
    ImageUtils.save_image(final_render, output_path)
    print(f"[OK] Imagen guardada en resolución: ({full_width}x{full_height}): {output_path}")


if __name__ == "__main__":
    main()