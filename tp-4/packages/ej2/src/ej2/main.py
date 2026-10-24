__all__ = ["main"]

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
    print("Current configuration:")
    pprint.pprint(cfg)
    dispatch(cfg)
