import sys
from tabulate import tabulate

from tp_5.config.parser import build_parser
from tp_5.config.loader import load_and_merge_config
from tp_5.config.validator import build_config_object


def print_config_table(config: dict) -> None:
    """Prints the config dictionary in table format"""
    table_data = [[key, str(value)] for key, value in sorted(config.items())]
    
    headers = ["Parametro", "Valor"]
    print("\n" + "=" * 50)
    print("      CONFIGURACION FINAL CARGADA (TP5)")
    print("=" * 50)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("=" * 50 + "\n")


def main() -> None:
    """Config test"""
    parser = build_parser()

    try:
        config = load_and_merge_config(parser)
        build_config_object(config)

    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"\n[ERROR]: {e}", file=sys.stderr)
        sys.exit(1)

    print_config_table(config)
    print("¡Configuración cargada y validada con éxito!")


if __name__ == "__main__":
    main()