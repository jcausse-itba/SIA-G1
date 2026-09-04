import sys
from tp_2.config.loader import load_and_merge_config
from tp_2.config.parser import build_parser
from tp_2.config.validator import validate_config
from tp_2.ga.engine import GAEngine
from tp_2.ga.fitness import FitnessEvaluator
from tp_2.image.render import render_individual
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

  # Soporte para la ruta de la imagen desde CLI o YAML
  image_path = getattr(cfg, "image_path", getattr(cfg, "image", None))
  if not image_path:
    image_path = "figures/germany-flag.png"

  output_path = getattr(cfg, "output_path", "output.png")

  print(f"Cargando imagen: {image_path}")

  # Resolución para evaluación rápida en el loop evolutivo (por defecto 128px)
  eval_size = getattr(cfg, "eval_size", 128)

  # Imagen en baja resolución para acelerar el cálculo de fitness
  target_img_eval = ImageUtils.load_target_image(
      image_path, max_size=(eval_size, eval_size)
  )

  # Imagen original completa para el renderizado final
  target_img_full = ImageUtils.load_target_image(image_path)

  evaluator = FitnessEvaluator(target_img_eval)
  engine = GAEngine(cfg, evaluator)

  best_individual = engine.run()

  # Renderizado final con la resolución nativa original
  final_render = render_individual(
      best_individual, target_img_full.shape[1], target_img_full.shape[0]
  )
  ImageUtils.save_image(final_render, output_path)


if __name__ == "__main__":
  main()