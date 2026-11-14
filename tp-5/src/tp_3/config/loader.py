import argparse
import json
import os
import sys
from typing import Any, Dict

import tomllib


def _flatten_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convierte recursivamente un diccionario con sub-diccionarios (como las secciones de TOML)
    en un único diccionario plano de un solo nivel.
    """
    flat = {}
    for key, value in d.items():
        if isinstance(value, dict):
            flat.update(_flatten_dict(value))
        else:
            flat[key] = value
    return flat


def _read_config_file(file_path: str) -> Dict[str, Any]:
    """Reads a configuration file using match/case based on its extension and flattens nested sections."""
    ext = os.path.splitext(file_path)[1].lower()

    match ext:
        case ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        case ".toml":
            with open(file_path, "rb") as f:
                data = tomllib.load(f)

        case _:
            raise ValueError(
                f"Unsupported configuration file format '{ext}'. "
                "Supported formats: .json, .toml"
            )

    # Aplanamos el diccionario para extraer todas las propiedades de las secciones [data], [model], etc.
    return _flatten_dict(data)


def load_and_merge_config(parser: argparse.ArgumentParser) -> Dict[str, Any]:
    """
    Parses CLI arguments and, if a JSON/TOML configuration file is provided,
    merges both giving priority to arguments explicitly passed via CLI.
    """
    cli_args = parser.parse_args()
    config = vars(cli_args)  # Convert Namespace to dictionary

    if cli_args.config:
        if not os.path.exists(cli_args.config):
            raise FileNotFoundError(f"Configuration file not found: {cli_args.config}")

        file_config = _read_config_file(cli_args.config)

        cli_supplied_keys = {
            action.dest for action in parser._actions
            if any(arg in sys.argv for arg in action.option_strings)
        }

        for key, val in file_config.items():
            if key not in cli_supplied_keys:
                config[key] = val

    return config