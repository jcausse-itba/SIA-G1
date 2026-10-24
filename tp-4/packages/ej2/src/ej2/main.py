import pprint
from pathlib import Path

from .cli import parse_config

DATA_PATH = Path("data/hopfield")


def main() -> None:
    cfg = parse_config(DATA_PATH)
    print("Current configuration:")
    pprint.pprint(cfg)  # TODO: despachar según cfg["task"]
