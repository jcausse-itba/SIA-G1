from tp_3.config.loader import load_and_merge_config
from tp_3.config.parser import build_parser
from tp_3.pipeline import run_pipeline


def main() -> None:
    parser = build_parser()
    try:
        config = load_and_merge_config(parser)
        run_pipeline(config)
    except (FileNotFoundError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()