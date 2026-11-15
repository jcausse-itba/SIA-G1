from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from functools import partial
import importlib
import inspect
import json
from pathlib import Path
import pickle
import pkgutil
import re
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

ALIASES: dict[str, str] = {
    "linear": "identity",
    "mse": "meansquarederror",
    "bce": "binarycrossentropy",
    "vae": "variationalautoencoder",
    "dae": "denoisingautoencoder",
    "ae": "autoencoder",
    "none": "nonoise",
    "h": "fontdataset",
    "csv": "csvdataset",
}


def resolve_class(package_path: str, name: str) -> type[Any]:
    """Dynamically locates and loads a class from a package (or module) by name/alias via reflection."""
    pkg = importlib.import_module(package_path)
    raw_target = name.lower().replace("_", "").lstrip(".")
    target = ALIASES.get(raw_target, raw_target)

    modules = [pkg]
    if hasattr(pkg, "__path__"):
        for _, modname, _ in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            modules.append(importlib.import_module(modname))

    for mod in modules:
        for cls_name, cls in inspect.getmembers(mod, inspect.isclass):
            if not inspect.isabstract(cls) and cls_name.lower() == target:
                return cls

    raise ValueError(f'Could not resolve class "{name}" in "{package_path}"')


def instantiate_with_reflection(
    cls: type[Any],
    config: dict[str, Any],
    injected_deps: dict[str, Any] | None = None,
) -> Any:
    """Instantiates `cls` by matching its constructor signature against injected dependencies and config keys."""
    sig = inspect.signature(cls.__init__)
    deps = injected_deps or {}
    param_map = {
        "alpha": "momentum_beta",
        "level": "noise_level",
    }

    kwargs: dict[str, Any] = {}
    for param_name in sig.parameters:
        if param_name == "self":
            continue
        if param_name in deps:
            kwargs[param_name] = deps[param_name]
        else:
            cfg_key = param_map.get(param_name, param_name)
            if cfg_key in config and config[cfg_key] is not None:
                kwargs[param_name] = config[cfg_key]

    return cls(**kwargs)


# =========================================================================
# POLYMORPHIC DATASET LOADERS (Resolved via Reflection by File Extension)
# =========================================================================
@dataclass(frozen=True)
class DatasetBundle:
    inputs: NDArray[np.float64]
    labels: list[str]
    image_shape: tuple[int, int] = (7, 5)


class BaseDataset(ABC):
    @abstractmethod
    def load(self, path: Path, subset_size: int | None = None) -> DatasetBundle:
        pass


class FontDataset(BaseDataset):
    """Parses C header `font.h` hex arrays into 32 binary (7x5 = 35) character patterns."""

    ROW_PATTERN = re.compile(r"\{([^}]+)\}")
    HEX_PATTERN = re.compile(r"0x[0-9a-fA-F]+")

    def load(self, path: Path, subset_size: int | None = None) -> DatasetBundle:
        text = path.read_text(encoding="utf-8")
        clean_text = "\n".join(line.split("//")[0] for line in text.splitlines())

        patterns: list[list[float]] = []
        for match in self.ROW_PATTERN.finditer(clean_text):
            hex_tokens = self.HEX_PATTERN.findall(match.group(1))
            if len(hex_tokens) == 7:
                bits = [
                    float((int(tok, 16) >> (4 - bit)) & 1)
                    for tok in hex_tokens
                    for bit in range(5)
                ]
                patterns.append(bits)

        if not patterns:
            raise ValueError(f"No valid 7-byte character patterns found in '{path}'.")

        X = np.array(patterns, dtype=np.float64)
        labels = ["DEL" if (0x60 + i) == 0x7F else chr(0x60 + i) for i in range(len(X))]

        if subset_size is not None:
            X = X[:subset_size]
            labels = labels[:subset_size]

        return DatasetBundle(inputs=X, labels=labels, image_shape=(7, 5))


class CsvDataset(BaseDataset):
    """Loads custom/emoji datasets from CSV files."""

    def load(self, path: Path, subset_size: int | None = None) -> DatasetBundle:
        df = pd.read_csv(path)
        label_col = "label" if "label" in df.columns else df.columns[0]
        labels = [str(v) for v in df[label_col].tolist()]

        if "image" in df.columns:
            rows = [v if isinstance(v, list) else json.loads(str(v)) for v in df["image"]]
            X = np.array(rows, dtype=np.float64)
        else:
            X = df.drop(columns=[label_col]).to_numpy(dtype=np.float64)

        if subset_size is not None:
            X = X[:subset_size]
            labels = labels[:subset_size]

        side = int(round(np.sqrt(X.shape[1])))
        shape = (side, side) if side * side == X.shape[1] else (7, 5)
        return DatasetBundle(inputs=X, labels=labels, image_shape=shape)


# =========================================================================
# DEPENDENCY INJECTION CONTAINER & CONFIGURATION OBJECT
# =========================================================================
@dataclass
class AutoencoderConfig:
    model_type: str = "autoencoder"
    dataset_path: Path = PROJECT_ROOT / "doc" / "font.h"
    subset_size: int | None = None

    encoder_layers: list[int] = field(default_factory=lambda: [35, 32, 16, 2])
    decoder_layers: list[int] | None = None
    hidden_activation: str = "tanh"
    latent_activation: str = "linear"
    output_activation: str = "sigmoid"
    beta: float = 1.0

    noise_type: str = "bit_flip"
    noise_level: float = 0.1
    eval_noise_levels: list[float] = field(default_factory=lambda: [0.05, 0.10, 0.20, 0.30])

    kl_loss: str = "kl_divergence"
    kl_weight: float = 0.001

    learning_rate: float = 0.005
    optimizer: str = "adam"
    momentum_beta: float = 0.9
    loss: str = "binary_cross_entropy"
    batch_size: int | None = 8
    max_epochs: int = 4000
    threshold: float = 0.5
    seed: int = 42

    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    save_model_path: Path | None = None
    load_model_path: Path | None = None
    output_dir: Path = PROJECT_ROOT / "results"

    def __post_init__(self) -> None:
        self.dataset_path = self._resolve_path(self.dataset_path)
        self.output_dir = self._resolve_path(self.output_dir)
        if self.save_model_path:
            self.save_model_path = self._resolve_path(self.save_model_path)
        if self.load_model_path:
            self.load_model_path = self._resolve_path(self.load_model_path)
        self.validate()

    @staticmethod
    def _resolve_path(p: str | Path) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    def validate(self) -> None:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {self.dataset_path}")
        if self.load_model_path and not self.load_model_path.exists():
            raise FileNotFoundError(f"Pre-trained model file not found at: {self.load_model_path}")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate (--learning-rate) must be positive.")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs (--max-epochs) must be a positive integer.")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size (--batch-size) must be greater than zero.")
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("threshold (--threshold) must be between 0.0 and 1.0.")
        if len(self.encoder_layers) < 2 or any(s <= 0 for s in self.encoder_layers):
            raise ValueError("encoder_layers must contain at least 2 positive integers.")

        # Validate that all configured class names can be resolved via reflection
        resolve_class("tp_5.engine.models", self.model_type)
        resolve_class("tp_5.engine.activation_functions", self.hidden_activation)
        resolve_class("tp_5.engine.activation_functions", self.latent_activation)
        resolve_class("tp_5.engine.activation_functions", self.output_activation)
        resolve_class("tp_5.engine.loss_functions", self.loss)
        resolve_class("tp_5.engine.optimizers", self.optimizer)
        resolve_class("tp_5.engine.noises", self.noise_type)
        resolve_class(__name__, self.dataset_path.suffix)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def build_dataset(self) -> DatasetBundle:
        """Resolves the dataset loader class via reflection on the file extension and loads data."""
        loader_cls = resolve_class(__name__, self.dataset_path.suffix)
        loader: BaseDataset = instantiate_with_reflection(loader_cls, self.to_dict())
        bundle = loader.load(self.dataset_path, self.subset_size)
        if bundle.inputs.shape[1] != self.encoder_layers[0]:
            raise ValueError(
                f"encoder_layers[0]={self.encoder_layers[0]} does not match dataset features ({bundle.inputs.shape[1]})."
            )
        return bundle

    def build_noise(self, override_level: float | None = None) -> Any:
        """Resolves and instantiates a Noise strategy via reflection."""
        noise_cls = resolve_class("tp_5.engine.noises", self.noise_type)
        cfg = self.to_dict()
        if override_level is not None:
            cfg["noise_level"] = override_level
        return instantiate_with_reflection(noise_cls, cfg)

    def build_optimizer_factory(self) -> Callable[..., Any]:
        """Resolves the Optimizer class via reflection and binds configured hyperparameters."""
        opt_cls = resolve_class("tp_5.engine.optimizers", self.optimizer)
        sig = inspect.signature(opt_cls.__init__)
        cfg = self.to_dict()
        param_map = {"alpha": "momentum_beta"}
        bound_kwargs = {
            p: cfg[param_map.get(p, p)]
            for p in sig.parameters
            if p not in ("self", "lr") and param_map.get(p, p) in cfg and cfg[param_map.get(p, p)] is not None
        }
        return partial(opt_cls, **bound_kwargs) if bound_kwargs else opt_cls

    def build_model(self) -> Any:
        """Resolves the target Model and injects all required collaborators via reflection."""
        np.random.seed(self.seed)
        if self.load_model_path:
            with self.load_model_path.open("rb") as f:
                checkpoint = pickle.load(f)
            return checkpoint.get("model", checkpoint)

        cfg = self.to_dict()
        make_act = lambda name: instantiate_with_reflection(
            resolve_class("tp_5.engine.activation_functions", name), cfg
        )
        make_loss = lambda name: instantiate_with_reflection(
            resolve_class("tp_5.engine.loss_functions", name), cfg
        )

        # Dependency Injection map: any constructor parameter matching these keys gets the injected instance
        injected_dependencies: dict[str, Any] = {
            "hidden_activation": make_act(self.hidden_activation),
            "input_activation": make_act(self.hidden_activation),
            "latent_activation": make_act(self.latent_activation),
            "output_activation": make_act(self.output_activation),
            "loss_function": make_loss(self.loss),
            "kl_loss": make_loss(self.kl_loss),
            "optimizer": self.build_optimizer_factory(),
            "noise_strategy": self.build_noise(),
        }

        model_cls = resolve_class("tp_5.engine.models", self.model_type)
        return instantiate_with_reflection(model_cls, cfg, injected_dependencies)


def build_config_object(raw_config: dict[str, Any]) -> AutoencoderConfig:
    """Builds and validates the AutoencoderConfig object dynamically matching constructor fields."""
    valid_fields = {
        name for name in inspect.signature(AutoencoderConfig.__init__).parameters if name != "self"
    }
    filtered = {k: v for k, v in raw_config.items() if k in valid_fields and v is not None}
    return AutoencoderConfig(**filtered)


def validate_config(cfg: dict[str, Any]) -> AutoencoderConfig:
    """Validates configuration dictionary and returns the dependency-injection config object."""
    return build_config_object(cfg)
