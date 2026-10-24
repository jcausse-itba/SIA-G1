__all__ = ["main"]

import json
import os
import pprint
from pathlib import Path
from typing import Dict, Any

from .cli import parse_config
from ej2.letters import run_letters_task
from ej2.qr import run_qr_task


DATA_PATH = Path("data/hopfield")

DISPATCHER_HANDLERS = {
    "qr": run_qr_task,
    "letters": run_letters_task,
    "all": lambda cfg: (
        run_letters_task(cfg),
        run_qr_task(cfg)
    )
}


def dispatch(cfg: Dict[str, Any]) -> None:
    handler = DISPATCHER_HANDLERS.get(cfg["task"])
    if handler is None:
        raise ValueError(f"Unknown task: {cfg['task']}")
    handler(cfg)


def main() -> None:
    cfg = parse_config(DATA_PATH)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(os.path.join(cfg["output_dir"], "config_usada.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False, default=str)
    print("Current configuration:")
    pprint.pprint(cfg)
    dispatch(cfg)
