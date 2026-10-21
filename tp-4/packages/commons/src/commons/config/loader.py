import argparse
import json
import os
import sys
from typing import Any, Dict, Sequence

import tomllib


def _flatten_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Aplana recursivamente las secciones del TOML/JSON ([data], [kohonen], ...)."""
    flat: Dict[str, Any] = {}
    for key, value in d.items():
        if isinstance(value, dict):
            for k, v in _flatten_dict(value).items():
                if k in flat:
                    raise ValueError(f"Duplicated key '{k}' across config sections.")
                flat[k] = v
        else:
            if key in flat:
                raise ValueError(f"Duplicated key '{key}' across config sections.")
            flat[key] = value
    return flat


def _read_config_file(file_path: str) -> Dict[str, Any]:
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
                f"Unsupported configuration file format '{ext}'. Supported formats: .json, .toml"
            )
    flat = _flatten_dict(data)
    # permite escribir 'grid-rows' o 'grid_rows' en el archivo
    return {k.replace("-", "_"): v for k, v in flat.items()}


def _explicit_keys(parser: argparse.ArgumentParser, argv: Sequence[str]) -> set[str]:
    """Dests realmente pasados por CLI (soporta '--x v', '--x=v', abreviaturas y '--no-x')."""
    saved = {id(a): a.default for a in parser._actions}
    saved_defaults = dict(parser._defaults)
    try:
        for a in parser._actions:
            a.default = argparse.SUPPRESS
        parser._defaults.clear()
        ns = parser.parse_args(list(argv))
    finally:
        for a in parser._actions:
            a.default = saved[id(a)]
        parser._defaults.update(saved_defaults)
    return set(vars(ns))


def _coerce(action: argparse.Action, key: str, value: Any) -> Any:
    """Aplica type y choices del parser a un valor que vino del archivo."""
    if value is None:
        return None
    conv = action.type
    try:
        if callable(conv):
            if action.nargs is not None and isinstance(value, (list, tuple)):
                value = [conv(v) for v in value]
            else:
                value = conv(value)
    except (argparse.ArgumentTypeError, ValueError, TypeError) as e:
        raise ValueError(f"Invalid value for '{key}' in config file: {value!r} ({e})") from e

    if action.choices is not None:
        vals = value if isinstance(value, list) else [value]
        bad = [v for v in vals if v not in action.choices]
        if bad:
            raise ValueError(
                f"Invalid value for '{key}' in config file: {bad!r} "
                f"(choose from {list(action.choices)})"
            )
    return value


def load_and_merge_config(
    parser: argparse.ArgumentParser, argv: Sequence[str] | None = None
) -> Dict[str, Any]:
    """Parsea la CLI y, si hay --config, mezcla con el archivo (la CLI explícita gana)."""
    argv = sys.argv[1:] if argv is None else list(argv)
    cli_args = parser.parse_args(argv)
    config = vars(cli_args)

    if not cli_args.config:
        return config
    if not os.path.exists(cli_args.config):
        raise FileNotFoundError(f"Configuration file not found: {cli_args.config}")

    file_config = _read_config_file(cli_args.config)
    actions = {a.dest: a for a in parser._actions if a.dest != "help"}

    unknown = sorted(set(file_config) - set(actions))
    if unknown:
        raise ValueError(f"Unknown keys in config file: {unknown}")

    explicit = _explicit_keys(parser, argv)
    for key, val in file_config.items():
        if key not in explicit:
            config[key] = _coerce(actions[key], key, val)
    return config
