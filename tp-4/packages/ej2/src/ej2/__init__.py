import pprint

from .cli import parse_config


def main() -> None:
    cfg = parse_config()
    pprint.pprint(cfg)  # TODO: despachar según cfg["task"]


if __name__ == "__main__":
    main()