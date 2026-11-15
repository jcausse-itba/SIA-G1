import sys
from tabulate import tabulate

from tp_3.config.parser import build_parser
from tp_3.config.loader import load_and_merge_config
from tp_3.config.validator import validate_config


def print_config_table(config: dict) -> None:
    """Prints the config dictionary in table format"""
    table_data = [[key, str(value)] for key, value in sorted(config.items())]
    
    headers = ["Parametro", "Valor"]
    print("\n" + "=" * 50)
    print("      CONFIGURACION FINAL CARGADA (TP3)")
    print("=" * 50)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("=" * 50 + "\n")


def main() -> None:
    """Config test"""
    parser = build_parser()

    try:
        config = load_and_merge_config(parser)
        validate_config(config)

    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"\n[ERROR]: {e}", file=sys.stderr)
        sys.exit(1)

    print_config_table(config)
    print("¡Configuración cargada y validada con éxito!")


if __name__ == "__main__":
    main()