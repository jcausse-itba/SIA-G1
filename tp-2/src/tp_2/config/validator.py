import os
from typing import Any, Dict


def validate_config(cfg: Dict[str, Any]) -> None:
    """Validates the consistency of the input configuration data."""
    if not os.path.exists(cfg["image_path"]):
        raise FileNotFoundError(f"Target image does not exist at path: {cfg['image_path']}")
    if cfg["pop_size"] <= 0:
        raise ValueError("Population size (--pop-size) must be positive.")
    if cfg["children_size"] <= 0:
        raise ValueError("Number of children (--children-size) must be positive.")
    if not (0.0 <= cfg["crossover_prob"] <= 1.0):
        raise ValueError("Crossover probability must be between 0.0 and 1.0.")
    if not (0.0 <= cfg["mutation_prob"] <= 1.0):
        raise ValueError("Mutation probability must be between 0.0 and 1.0.")