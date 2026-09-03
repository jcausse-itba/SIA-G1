import sys

from tp_2.config.parser import build_parser
from tp_2.config.loader import load_and_merge_config
from tp_2.config.validator import validate_config

def main() -> None:
    parser = build_parser()

    try:
        cfg = load_and_merge_config(parser)
        validate_config(cfg)
    except Exception as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n[OK] Configuration parsed and validated successfully.")


if __name__ == "__main__":
    main()