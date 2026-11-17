"""Estudio del VAE sobre código de fractales — ejercicio 2.

Hace todo el recorrido, igual que study/ej1 y study/ej1b:

1. Genera el dataset con code_dataset/generator.py si todavía no existe.
2. Busca hiperparámetros con Optuna (TPE + MedianPruner). Cuanto más grande es la
   red del trial, más programas usa para entrenar.
3. Reentrena la mejor configuración con más épocas.
4. Evalúa la reconstrucción y la generación compilando los programas con
   Flex-Bison-Compiler y escribe gráficos, galerías, CSV y el modelo.

Cortar la ejecución (como en los otros TPs):
- Ctrl+C (una vez) o Shift+Q: termina el trial actual (o el entrenamiento final),
  no lanza más trials y genera igual todos los resultados con lo que haya.
- Ctrl+C (dos veces): aborta en el momento; se escriben los resultados de la
  búsqueda con los trials ya terminados.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import plotly.graph_objects as go
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from plotly.subplots import make_subplots

from tp_5.config.validator import (
    VOCAB_MAX_IDX,
    AutoencoderConfig,
    CodeDataset,
    DatasetBundle,
    build_config_object,
)
from tp_5.latent_explorer import FractalCompilerService


# =========================================================================
# CONSTANTES DEL ESTUDIO
# =========================================================================
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "study" / "ej2" / "resultados"
CODE_DATASET_DIR = ROOT / "code_dataset"
CODE_DIR = CODE_DATASET_DIR / "code"
GENERATOR = CODE_DATASET_DIR / "generator.py"

INPUT_DIM = 1000  # MAX_CODE_LEN: un carácter por posición
SEED = 42

# Cuerpos de red (capas ocultas del encoder; el decoder es el espejo).
BODIES: dict[str, list[int]] = {
    "256-128": [256, 128],
    "512-256": [512, 256],
    "512-512-256-256": [512, 512, 256, 256],
    "512x4-256": [512, 512, 512, 512, 256],
}
LATENT_DIMS = (8, 16, 32)
HIDDEN_ACTIVATIONS = ("tanh", "relu")
OPTIMIZERS = ("adam", "rmsprop", "momentum", "sgd")
LR_SCHEDULERS = ("constant", "cosine_annealing", "exponential_decay", "linear_decay")
BATCH_SIZES = (8, 16, 32)
LR_RANGE = (1e-4, 1e-2)
KL_WEIGHT_RANGE = (1e-6, 1e-2)
MIN_LR_RATIO_RANGE = (0.005, 0.2)
DECAY_RATE_RANGE = (0.01, 0.2)
MOMENTUM_RANGE = (0.8, 0.99)
BETA1_RANGE = (0.8, 0.99)
BETA2_RANGE = (0.95, 0.9999)
EPSILON_RANGE = (1e-8, 1e-6)

# El dataset crece con la red: la red más chica (256-128, latente 8) entrena con
# BASE_SAMPLES programas y el resto con una cantidad proporcional a sus parámetros.
BASE_SAMPLES = 100
MAX_SAMPLES = 1000
SAMPLES_ROUND = 50

# Presupuesto
N_TRIALS = 60
TRIAL_EPOCHS = 1500  # fijo: los schedulers dependen del total de épocas
FINAL_EPOCHS = 3000
TIME_PER_TRIAL_BASE_SEC = 300.0  # por cada BASE_SAMPLES programas (escala con el dataset)
EVAL_EVERY = 50  # épocas entre reportes intermedios (para el pruner)
LOG_EVERY = 20  # épocas entre puntos de la curva final

# Evaluación (compilando con Flex-Bison-Compiler)
TRIAL_COMPILE_SAMPLES = 60  # reconstrucciones compiladas por trial
TRIAL_GEN_SAMPLES = 40  # muestras generadas por trial (prior y local)
FINAL_PRIOR_SAMPLES = 200
FINAL_LOCAL_SAMPLES = 300
LOCAL_SCALES = (0.25, 0.5, 1.0)  # z = μᵢ + k·σᵢ·ε
LOCAL_SCALE_TRIAL = 0.25
INTERPOLATION_STEPS = 7
COMPILE_WORKERS = 4
GALLERY_PER_FAMILY = 4

# Objetivo a minimizar (todos los términos están en [0, 1])
SCORE_WEIGHTS = {
    "rec_compile": 0.40,  # programas reconstruidos que compilan
    "char_acc": 0.30,  # caracteres reconstruidos correctamente
    "local_novel": 0.15,  # muestras locales nuevas que compilan
    "prior_valid": 0.15,  # muestras de N(0, I) que compilan
}

FAMILY_COLORS = {
    "Geométrico recursivo": "#2A3990",
    "Tiempo de escape": "#D23369",
    "IFS": "#2A9D8F",
    "Híbrido": "#E0A100",
}
NAVY, NAVY2, PINK, GREY, TEAL, GOLD = "#2A3990", "#3949AB", "#D23369", "#999999", "#2A9D8F", "#E0A100"
PLOT_LAYOUT = dict(template="plotly_white", font=dict(family="Arial", size=13), title_x=0.02)


# =========================================================================
# CORTE SEGURO: Ctrl+C y Shift+Q
# =========================================================================
class StopController:
    """Primer Ctrl+C (o Shift+Q): cortar al terminar el trial. Segundo Ctrl+C: abortar."""

    def __init__(self) -> None:
        self.requested = False
        self._presses = 0
        self._running = False
        self._listener: threading.Thread | None = None
        self._previous_handler: Any = None

    def install(self) -> None:
        self._previous_handler = signal.signal(signal.SIGINT, self._on_sigint)
        self._running = True
        self._listener = threading.Thread(target=self._listen_for_quit, daemon=True)
        self._listener.start()
        print("\n[INFO] Ctrl+C o Shift+Q: termina el trial actual y genera los resultados.")
        print("[INFO] Ctrl+C dos veces: aborta en el momento.")

    def uninstall(self) -> None:
        self._running = False
        if self._listener is not None:
            self._listener.join(timeout=0.5)
        signal.signal(signal.SIGINT, self._previous_handler or signal.default_int_handler)

    def rearm(self, message: str) -> None:
        """Permite cortar una etapa nueva (por ejemplo, el entrenamiento final)."""
        self.requested = False
        self._presses = 0
        signal.signal(signal.SIGINT, self._on_sigint)
        print(f"\n[INFO] {message}")

    def _request(self, source: str) -> None:
        if not self.requested:
            self.requested = True
            print(f"\n[!] {source} detectado. Se termina lo actual y se generan los resultados...")

    def _on_sigint(self, signum: int, frame: Any) -> None:
        self._presses += 1
        if self._presses == 1:
            self._request("Ctrl+C")
            print("    (Ctrl+C otra vez para abortar en el momento)")
        else:
            signal.signal(signal.SIGINT, signal.default_int_handler)
            raise KeyboardInterrupt

    def _listen_for_quit(self) -> None:
        """Igual que el TP3: escucha Shift+Q ('Q') en Windows y en Unix."""
        try:
            import msvcrt  # type: ignore[import-not-found]

            while self._running:
                if msvcrt.kbhit() and msvcrt.getch() == b"Q":
                    self._request("Shift+Q")
                    break
                time.sleep(0.1)
        except ImportError:
            import select

            fd = None
            old_settings = None
            try:
                import termios
                import tty

                if not sys.stdin.isatty():
                    return
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                tty.setcbreak(fd)  # cbreak mantiene Ctrl+C como señal
            except Exception:
                return
            try:
                while self._running:
                    if select.select([sys.stdin], [], [], 0.2)[0] and sys.stdin.read(1) == "Q":
                        self._request("Shift+Q")
                        break
            finally:
                if old_settings is not None:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


STOP = StopController()


# =========================================================================
# DATASET
# =========================================================================
def ensure_code_dataset() -> None:
    """Corre code_dataset/generator.py si no existe la carpeta code/ (genera 10 000 programas)."""
    if CODE_DIR.is_dir() and any(CODE_DIR.iterdir()):
        return
    print(f"[INFO] No existe {CODE_DIR}. Generando programas con {GENERATOR.name}...")
    subprocess.run([sys.executable, str(GENERATOR)], cwd=CODE_DATASET_DIR, check=True)


def family_of(label: str) -> str:
    if "hybrid" in label:
        return "Híbrido"
    if "ifs" in label:
        return "IFS"
    if any(key in label for key in ("julia", "mandelbrot", "multibrot", "biomorph")):
        return "Tiempo de escape"
    return "Geométrico recursivo"


class DatasetCache:
    """Carga una vez los MAX_SAMPLES primeros programas y entrega prefijos del tamaño pedido."""

    def __init__(self) -> None:
        cfg = base_config(subset_size=MAX_SAMPLES)
        self.full: DatasetBundle = cfg.build_dataset()
        self.texts = [CodeDataset.decode_vector(row) for row in self.full.inputs]
        self.families = [family_of(label) for label in self.full.labels]

    def take(self, n: int) -> tuple[np.ndarray, list[str], list[str], list[str]]:
        n = min(n, len(self.full.inputs))
        return self.full.inputs[:n], self.full.labels[:n], self.texts[:n], self.families[:n]


# =========================================================================
# ARQUITECTURA Y CONFIGURACIÓN
# =========================================================================
def encoder_layers(body: str, latent_dim: int) -> list[int]:
    return [INPUT_DIM, *BODIES[body], latent_dim]


def decoder_layers(body: str) -> list[int]:
    return [*reversed(BODIES[body]), INPUT_DIM]


def parameter_count(body: str, latent_dim: int) -> int:
    enc = encoder_layers(body, latent_dim)
    dec = [latent_dim, *decoder_layers(body)]
    count = 0
    for i in range(len(enc) - 1):
        out = enc[i + 1] * (2 if i == len(enc) - 2 else 1)  # la capa latente produce μ y σ
        count += (enc[i] + 1) * out
    for i in range(len(dec) - 1):
        count += (dec[i] + 1) * dec[i + 1]
    return count


REFERENCE_PARAMS = parameter_count(next(iter(BODIES)), min(LATENT_DIMS))


def dataset_size_for(body: str, latent_dim: int) -> int:
    """Programas de entrenamiento proporcionales a los parámetros de la red."""
    raw = BASE_SAMPLES * parameter_count(body, latent_dim) / REFERENCE_PARAMS
    rounded = int(round(raw / SAMPLES_ROUND) * SAMPLES_ROUND)
    return int(min(MAX_SAMPLES, max(BASE_SAMPLES, rounded)))


def base_config(**overrides: Any) -> AutoencoderConfig:
    raw: dict[str, Any] = {
        "model_type": "variational_autoencoder",
        "dataset_path": CODE_DIR,
        "subset_size": BASE_SAMPLES,
        "encoder_layers": encoder_layers("256-128", 16),
        "decoder_layers": decoder_layers("256-128"),
        "hidden_activation": "tanh",
        "latent_activation": "linear",
        "output_activation": "linear",
        "loss": "mean_squared_error",
        "kl_loss": "kl_divergence",
        "kl_weight": 1e-5,
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "lr_scheduler": "constant",
        "batch_size": 16,
        "max_epochs": TRIAL_EPOCHS,
        "residual": True,
        "seed": SEED,
        "output_dir": OUTPUT_DIR,
    }
    raw.update(overrides)
    return build_config_object(raw)


def config_from_params(params: dict[str, Any], n_samples: int, max_epochs: int, seed: int) -> AutoencoderConfig:
    body, latent = params["body"], params["latent_dim"]
    overrides: dict[str, Any] = {
        "subset_size": n_samples,
        "encoder_layers": encoder_layers(body, latent),
        "decoder_layers": decoder_layers(body),
        "hidden_activation": params["hidden_activation"],
        "residual": params["residual"],
        "kl_weight": params["kl_weight"],
        "optimizer": params["optimizer"],
        "learning_rate": params["learning_rate"],
        "lr_scheduler": params["lr_scheduler"],
        "batch_size": params["batch_size"],
        "max_epochs": max_epochs,
        "seed": seed,
    }
    for key in ("min_lr_ratio", "decay_rate", "momentum_beta", "beta1", "beta2", "epsilon"):
        if key in params:
            overrides[key] = params[key]
    return base_config(**overrides)


def sample_params(trial: optuna.Trial) -> dict[str, Any]:
    params: dict[str, Any] = {
        "body": trial.suggest_categorical("body", list(BODIES)),
        "latent_dim": trial.suggest_categorical("latent_dim", list(LATENT_DIMS)),
        "residual": trial.suggest_categorical("residual", [True, False]),
        "hidden_activation": trial.suggest_categorical("hidden_activation", list(HIDDEN_ACTIVATIONS)),
        "kl_weight": trial.suggest_float("kl_weight", *KL_WEIGHT_RANGE, log=True),
        "optimizer": trial.suggest_categorical("optimizer", list(OPTIMIZERS)),
        "learning_rate": trial.suggest_float("learning_rate", *LR_RANGE, log=True),
        "lr_scheduler": trial.suggest_categorical("lr_scheduler", list(LR_SCHEDULERS)),
        "batch_size": trial.suggest_categorical("batch_size", list(BATCH_SIZES)),
    }
    if params["lr_scheduler"] in ("cosine_annealing", "linear_decay"):
        params["min_lr_ratio"] = trial.suggest_float("min_lr_ratio", *MIN_LR_RATIO_RANGE, log=True)
    if params["lr_scheduler"] == "exponential_decay":
        params["decay_rate"] = trial.suggest_float("decay_rate", *DECAY_RATE_RANGE, log=True)
    if params["optimizer"] == "momentum":
        params["momentum_beta"] = trial.suggest_float("momentum_beta", *MOMENTUM_RANGE)
    if params["optimizer"] == "adam":
        params["beta1"] = trial.suggest_float("beta1", *BETA1_RANGE)
        params["beta2"] = trial.suggest_float("beta2", *BETA2_RANGE)
        params["epsilon"] = trial.suggest_float("epsilon", *EPSILON_RANGE, log=True)
    return params


# =========================================================================
# MÉTRICAS
# =========================================================================
def to_indices(values: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.asarray(values) * VOCAB_MAX_IDX), 0, int(VOCAB_MAX_IDX)).astype(int)


def char_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """Caracteres reales (sin el relleno) que vuelven a su índice exacto."""
    t, p = to_indices(target), to_indices(pred)
    mask = t != 0
    return float(np.mean(t[mask] == p[mask]))


def decode_texts(matrix: np.ndarray) -> list[str]:
    return [CodeDataset.decode_vector(row) for row in matrix]


def compile_many(compiler: FractalCompilerService, codes: list[str]) -> list[dict[str, Any]]:
    with ThreadPoolExecutor(COMPILE_WORKERS) as pool:
        return list(pool.map(compiler.compile_code, codes))


class EvalMode:
    """Apaga el muestreo de la capa variacional mientras se evalúa (z = μ)."""

    def __init__(self, model: Any) -> None:
        self.layer = getattr(model, "variational_layer", None)
        self.previous = False

    def __enter__(self) -> "EvalMode":
        if self.layer is not None:
            self.previous = self.layer.training
            self.layer.training = False
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.layer is not None:
            self.layer.training = self.previous


def latent_stats(model: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with EvalMode(model):
        mu = model.encode(X)
        layer = getattr(model, "variational_layer", None)
        sigma = layer.std.copy() if layer is not None else np.zeros_like(mu)
    return mu, sigma


def sample_rates(compiled: list[dict[str, Any]], texts: list[str], known: set[str]) -> dict[str, float]:
    ok = np.array([bool(r["ok"]) for r in compiled])
    copy = np.array([t in known for t in texts])
    n = max(1, len(ok))
    return {
        "valid": float(ok.mean()) if len(ok) else 0.0,
        "copy": float((ok & copy).sum() / n),
        "novel": float((ok & ~copy).sum() / n),
    }


def evaluate_model(
    model: Any,
    X: np.ndarray,
    texts: list[str],
    compiler: FractalCompilerService,
    rng: np.random.Generator,
    n_compile: int,
    n_prior: int,
    n_local: int,
    local_scales: tuple[float, ...] = (LOCAL_SCALE_TRIAL,),
    keep_samples: bool = False,
) -> dict[str, Any]:
    known = set(texts)
    with EvalMode(model):
        pred = model.forward(X)
    out: dict[str, Any] = {
        "char_acc": char_accuracy(pred, X),
        "mse": float(np.mean((pred - X) ** 2)),
    }
    idx = rng.choice(len(X), size=min(n_compile, len(X)), replace=False)
    rec_texts = decode_texts(pred[idx])
    rec_compiled = compile_many(compiler, rec_texts)
    out["rec_compile"] = float(np.mean([r["ok"] for r in rec_compiled]))

    mu, sigma = latent_stats(model, X)
    out["latent_abs_mean"] = float(np.abs(mu.mean(0)).mean())
    out["sigma_mean"] = float(sigma.mean())

    with EvalMode(model):
        z_prior = rng.standard_normal((n_prior, mu.shape[1]))
        prior_texts = decode_texts(model.decode(z_prior))
    prior_compiled = compile_many(compiler, prior_texts)
    prior = sample_rates(prior_compiled, prior_texts, known)
    out.update({f"prior_{k}": v for k, v in prior.items()})

    samples: dict[str, Any] = {"prior": list(zip(prior_texts, prior_compiled))}
    for k in local_scales:
        base = rng.integers(0, len(X), n_local)
        z_local = mu[base] + k * sigma[base] * rng.standard_normal((n_local, mu.shape[1]))
        with EvalMode(model):
            local_texts = decode_texts(model.decode(z_local))
        local_compiled = compile_many(compiler, local_texts)
        rates = sample_rates(local_compiled, local_texts, known)
        out.update({f"local_{k}_{name}": v for name, v in rates.items()})
        samples[f"local_{k}"] = list(zip(local_texts, local_compiled, base.tolist()))
    if keep_samples:
        out["_samples"] = samples
    return out


def objective_score(metrics: dict[str, Any]) -> float:
    w = SCORE_WEIGHTS
    return float(
        w["rec_compile"] * (1 - metrics["rec_compile"])
        + w["char_acc"] * (1 - metrics["char_acc"])
        + w["local_novel"] * (1 - metrics[f"local_{LOCAL_SCALE_TRIAL}_novel"])
        + w["prior_valid"] * (1 - metrics["prior_valid"])
    )


# =========================================================================
# BÚSQUEDA CON OPTUNA
# =========================================================================
def make_objective(cache: DatasetCache, compiler: FractalCompilerService):
    def objective(trial: optuna.Trial) -> float:
        if STOP.requested:
            raise optuna.TrialPruned("Corte pedido por el usuario")
        params = sample_params(trial)
        n_samples = dataset_size_for(params["body"], params["latent_dim"])
        X, _, texts, _ = cache.take(n_samples)
        trial.set_user_attr("n_samples", int(len(X)))
        trial.set_user_attr("n_params", parameter_count(params["body"], params["latent_dim"]))

        cfg = config_from_params(params, len(X), TRIAL_EPOCHS, SEED + trial.number)
        model = cfg.build_model(quantization_scale=VOCAB_MAX_IDX)
        time_limit = TIME_PER_TRIAL_BASE_SEC * len(X) / BASE_SAMPLES
        start = time.time()
        print(
            f"\n>>> Trial #{trial.number} | Red={params['body']} latente={params['latent_dim']} "
            f"residual={params['residual']} | Dataset={len(X)} programas | Opt={params['optimizer']} "
            f"η={params['learning_rate']:.5f} sched={params['lr_scheduler']} | batch={params['batch_size']} "
            f"| β={params['kl_weight']:.1e}"
        )

        def epoch_callback(epoch: int, current: Any) -> bool:
            if STOP.requested:
                return True
            if time.time() - start >= time_limit:
                trial.set_user_attr("time_limited", True)
                print(f"    [Tiempo] Límite de {time_limit:.0f}s alcanzado en la época {epoch}.")
                return True
            if epoch % EVAL_EVERY == 0:
                with EvalMode(current):
                    acc = char_accuracy(current.forward(X), X)
                trial.report(1.0 - acc, epoch)
                if trial.should_prune():
                    print(f"    [Pruned] Trial #{trial.number} en la época {epoch} (caracteres OK: {acc:.1%}).")
                    raise optuna.TrialPruned()
            return False

        model.fit(
            X, X, batch_size=cfg.batch_size, epochs=TRIAL_EPOCHS, lr=cfg.learning_rate,
            print_every=max(250, TRIAL_EPOCHS // 4), epoch_callback=epoch_callback,
        )
        if STOP.requested:
            raise optuna.TrialPruned("Corte pedido por el usuario")
        if not np.all(np.isfinite(model.forward(X))):
            raise optuna.TrialPruned("La red divergió (NaN/inf)")

        metrics = evaluate_model(
            model, X, texts, compiler, np.random.default_rng(SEED + trial.number),
            n_compile=TRIAL_COMPILE_SAMPLES, n_prior=TRIAL_GEN_SAMPLES, n_local=TRIAL_GEN_SAMPLES,
        )
        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        score = objective_score(metrics)
        print(
            f"    [Trial #{trial.number}] score={score:.4f} | caracteres OK={metrics['char_acc']:.1%} | "
            f"compilan={metrics['rec_compile']:.0%} | N(0,I) válidas={metrics['prior_valid']:.0%} | "
            f"locales nuevas={metrics[f'local_{LOCAL_SCALE_TRIAL}_novel']:.0%} | {time.time() - start:.0f}s"
        )
        return score

    return objective


def run_search(
    cache: DatasetCache, compiler: FractalCompilerService, n_trials: int, resume: bool
) -> tuple[optuna.Study, bool]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    db = OUTPUT_DIR / "optuna.db"
    if db.exists() and not resume:
        db.unlink()
    study = optuna.create_study(
        study_name="ej2_vae_fractales",
        storage=f"sqlite:///{db.as_posix()}",
        load_if_exists=resume,
        direction="minimize",
        sampler=TPESampler(seed=SEED),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=EVAL_EVERY * 4, interval_steps=EVAL_EVERY),
    )

    def stop_callback(st: optuna.Study, _trial: optuna.trial.FrozenTrial) -> None:
        if STOP.requested:
            st.stop()

    print("\n=== Búsqueda de hiperparámetros con Optuna (TPE + MedianPruner) ===")
    aborted = False
    try:
        study.optimize(
            make_objective(cache, compiler), n_trials=n_trials, callbacks=[stop_callback],
            catch=(Exception,), gc_after_trial=True,
        )
    except KeyboardInterrupt:
        aborted = True
        print("\n[!] Búsqueda abortada. Se generan los resultados con los trials terminados.")
    return study, aborted


def completed_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]


# =========================================================================
# GRÁFICOS DE OPTUNA (los mismos que el TP3)
# =========================================================================
def write_optuna_plots(study: optuna.Study) -> list[Path]:
    from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice

    written: list[Path] = []
    study.trials_dataframe().to_csv(OUTPUT_DIR / "optuna_trials.csv", index=False)
    written.append(OUTPUT_DIR / "optuna_trials.csv")
    plots = [
        ("plot_optimization_history.html", plot_optimization_history, 1),
        ("plot_param_importances.html", plot_param_importances, 2),
        ("plot_slice.html", plot_slice, 1),
    ]
    for name, fn, min_trials in plots:
        if len(completed_trials(study)) < min_trials:
            print(f"[WARN] {name}: hacen falta al menos {min_trials} trials completos.")
            continue
        try:
            fn(study).write_html(str(OUTPUT_DIR / name))
            written.append(OUTPUT_DIR / name)
        except Exception as exc:  # p. ej. importancias con un solo valor por parámetro
            print(f"[WARN] No se pudo generar {name}: {exc}")
    return written


# =========================================================================
# GRÁFICOS DEL ESTUDIO
# =========================================================================
def write_exploration_chart(study: optuna.Study) -> Path:
    trials = completed_trials(study)
    fig = make_subplots(
        1, 2, column_widths=[0.45, 0.55],
        subplot_titles=("Mejor trial por red", "Score vs. tamaño de red (tamaño del punto = programas)"),
    )
    bodies = list(BODIES)
    best = {b: min((t for t in trials if t.params.get("body") == b), key=lambda t: t.value, default=None) for b in bodies}
    labels = [f"{b}<br>{dataset_size_for(b, 16)} prog. (lat. 16)" for b in bodies]
    fig.add_bar(
        x=labels, y=[(1 - best[b].user_attrs.get("rec_compile", 0)) if best[b] else None for b in bodies],
        name="1 − compilan", marker_color=PINK, row=1, col=1,
    )
    fig.add_bar(
        x=labels, y=[(1 - best[b].user_attrs.get("char_acc", 0)) if best[b] else None for b in bodies],
        name="1 − caracteres OK", marker_color=NAVY, row=1, col=1,
    )
    for i, b in enumerate(bodies):
        ts = [t for t in trials if t.params.get("body") == b]
        if not ts:
            continue
        fig.add_scatter(
            x=[t.user_attrs.get("n_params", 0) for t in ts], y=[t.value for t in ts], mode="markers", name=b,
            marker=dict(size=[6 + t.user_attrs.get("n_samples", BASE_SAMPLES) / 40 for t in ts], opacity=0.75,
                        color=[NAVY, TEAL, GOLD, PINK][i % 4], line=dict(color="white", width=0.5)),
            text=[f"Trial {t.number}<br>{t.user_attrs.get('n_samples')} programas<br>latente {t.params.get('latent_dim')}"
                  for t in ts],
            hovertemplate="%{text}<br>parámetros=%{x:,}<br>score=%{y:.4f}<extra></extra>", row=1, col=2,
        )
    fig.update_yaxes(title_text="error (menor es mejor)", row=1, col=1)
    fig.update_xaxes(title_text="parámetros de la red", type="log", row=1, col=2)
    fig.update_yaxes(title_text="score (menor es mejor)", row=1, col=2)
    fig.update_layout(title="Exploración de arquitecturas (el dataset crece con la red)", barmode="group", **PLOT_LAYOUT)
    path = OUTPUT_DIR / "exploracion_arquitecturas.html"
    fig.write_html(path)
    return path


def write_architecture_diagram(params: dict[str, Any], n_samples: int) -> Path:
    enc = encoder_layers(params["body"], params["latent_dim"])
    dec = decoder_layers(params["body"])
    blocks = (
        [(str(n), NAVY if i else "#7890CD") for i, n in enumerate(enc[:-1])]
        + [(f"μ, σ ({enc[-1]})", NAVY2), (f"z = μ + σ·ε ({enc[-1]})", PINK)]
        + [(str(n), NAVY2 if i < len(dec) - 1 else "#7890CD") for i, n in enumerate(dec)]
    )
    fig = go.Figure()
    for i, (text, color) in enumerate(blocks):
        size = int(text) if text.isdigit() else 120
        h = 0.25 + 0.6 * size / INPUT_DIM
        fig.add_shape(type="rect", x0=i, x1=i + 0.8, y0=-h, y1=h, fillcolor=color, line=dict(width=0))
        fig.add_annotation(x=i + 0.4, y=0, text=f"<b>{text}</b>", showarrow=False, font=dict(color="white", size=11))
        if i < len(blocks) - 1:
            fig.add_annotation(x=i + 1, y=0, ax=i + 0.8, ay=0, xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=2, arrowcolor=GREY)
    k = len(enc) - 1
    fig.add_shape(type="rect", x0=k - 0.15, x1=k + 1.95, y0=-1.0, y1=1.0, line=dict(color=PINK, dash="dot"))
    fig.add_annotation(x=k + 0.9, y=1.1, text="reparametrización (ε ~ N(0, I))", showarrow=False, font=dict(color=PINK))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, range=[-1.3, 1.3])
    res = "con conexiones residuales" if params["residual"] else "sin conexiones residuales"
    fig.update_layout(
        title=f"Arquitectura elegida: {params['hidden_activation']}, {res}, salida lineal · {n_samples} programas",
        height=380, **PLOT_LAYOUT,
    )
    path = OUTPUT_DIR / "arquitectura_propuesta.html"
    fig.write_html(path)
    return path


def write_training_curves(log: list[tuple[int, float, float]], beta: float) -> Path:
    epochs = [e for e, _, _ in log]
    fig = go.Figure()
    fig.add_scatter(x=epochs, y=[r + beta * k for _, r, k in log], name="Total", line=dict(color=NAVY, width=2.5))
    fig.add_scatter(x=epochs, y=[r for _, r, _ in log], name="Reconstrucción (MSE)", line=dict(color=NAVY2, dash="dot"))
    fig.add_scatter(x=epochs, y=[beta * k for _, _, k in log], name="β·KL", line=dict(color=PINK))
    fig.update_yaxes(type="log", exponentformat="e", title="pérdida")
    fig.update_xaxes(title="época")
    fig.update_layout(title=f"Entrenamiento final · L = MSE + β·KL (β = {beta:.1e})", **PLOT_LAYOUT)
    path = OUTPUT_DIR / "entrenamiento_final.html"
    fig.write_html(path)
    return path


def write_latent_plot(mu: np.ndarray, labels: list[str], families: list[str]) -> Path:
    centered = mu - mu.mean(0)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[:2].T if mu.shape[1] > 2 else centered
    var = (s**2 / (s**2).sum())[:2] if mu.shape[1] > 2 else np.array([np.nan, np.nan])
    fig = go.Figure()
    for fam, color in FAMILY_COLORS.items():
        idx = [i for i, f in enumerate(families) if f == fam]
        fig.add_scatter(
            x=proj[idx, 0], y=proj[idx, 1], mode="markers", name=fam, text=[labels[i] for i in idx],
            marker=dict(color=color, size=9, line=dict(color="white", width=0.5)),
            hovertemplate="%{text}<extra></extra>",
        )
    fig.update_xaxes(title=f"PC1 ({var[0] * 100:.0f} % var.)")
    fig.update_yaxes(title=f"PC2 ({var[1] * 100:.0f} % var.)")
    fig.update_layout(title=f"Espacio latente ({mu.shape[1]} dim) proyectado con PCA · {len(mu)} programas", **PLOT_LAYOUT)
    path = OUTPUT_DIR / "espacio_latente_pca.html"
    fig.write_html(path)
    return path


def write_generation_chart(metrics: dict[str, Any]) -> Path:
    modes = [("N(0, I)", "prior")] + [(f"μᵢ + {k}·σᵢ·ε", f"local_{k}") for k in LOCAL_SCALES]
    fig = go.Figure()
    fig.add_bar(y=[m for m, _ in modes], x=[metrics[f"{k}_copy"] * 100 for _, k in modes], name="Copias exactas",
                orientation="h", marker_color=NAVY2)
    fig.add_bar(y=[m for m, _ in modes], x=[metrics[f"{k}_novel"] * 100 for _, k in modes], name="Programas nuevos",
                orientation="h", marker_color=PINK)
    fig.update_layout(
        barmode="stack", title=f"Muestras que compilan según cómo se elige z ({FINAL_PRIOR_SAMPLES} del prior, "
                               f"{FINAL_LOCAL_SAMPLES} por escala local)",
        xaxis=dict(title="% de las muestras", range=[0, 100]), yaxis=dict(autorange="reversed"), **PLOT_LAYOUT,
    )
    path = OUTPUT_DIR / "generacion.html"
    fig.write_html(path)
    return path


def _img(src: str | None, caption: str, highlight: bool = False) -> str:
    border = PINK if highlight else "#DDDDDD"
    body = (f'<img src="{src}" alt="{html.escape(caption)}">' if src
            else '<div class="fail">no compila</div>')
    return f'<figure style="border-color:{border}">{body}<figcaption>{html.escape(caption)}</figcaption></figure>'


def write_gallery(path: Path, title: str, sections: list[tuple[str, list[str]]], extra_html: str = "") -> Path:
    css = (
        "body{font-family:Arial,sans-serif;margin:24px;color:#333}h1{color:#2A3990}h2{color:#3949AB;margin-top:28px}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px}"
        "figure{margin:0;border:3px solid;border-radius:6px;overflow:hidden;background:#fafafa}"
        "img{width:100%;display:block}figcaption{font-size:12px;padding:6px;word-break:break-all}"
        ".fail{height:135px;display:flex;align-items:center;justify-content:center;color:#9C254D;background:#f4f4f4}"
        "pre{background:#f6f6f6;padding:8px;font-size:11px;white-space:pre-wrap}"
    )
    parts = [f"<!DOCTYPE html><html lang='es'><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
             f"<style>{css}</style></head><body><h1>{html.escape(title)}</h1>"]
    for heading, items in sections:
        parts.append(f"<h2>{html.escape(heading)}</h2><div class='grid'>{''.join(items)}</div>")
    parts.append(extra_html + "</body></html>")
    path.write_text("".join(parts), encoding="utf-8")
    return path


def save_bmp(result: dict[str, Any], path: Path) -> None:
    src = result.get("image_b64")
    if result.get("ok") and src:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(src.split(",", 1)[1]))


# =========================================================================
# ENTRENAMIENTO FINAL
# =========================================================================
def train_final(params: dict[str, Any], X: np.ndarray, epochs: int) -> tuple[Any, list[tuple[int, float, float]]]:
    cfg = config_from_params(params, len(X), epochs, SEED)
    model = cfg.build_model(quantization_scale=VOCAB_MAX_IDX)
    layer = model.variational_layer
    log: list[tuple[int, float, float]] = []

    def epoch_callback(epoch: int, current: Any) -> bool:
        if STOP.requested:
            print(f"    [Corte] Entrenamiento final detenido en la época {epoch}.")
            return True
        if epoch % LOG_EVERY == 0:
            with EvalMode(current):
                pred = current.forward(X)
                log.append((epoch, float(np.mean((pred - X) ** 2)), float(layer.kl_loss.compute(layer.mu, layer.std))))
        return False

    print(f"\n=== Entrenamiento final: {epochs} épocas con {len(X)} programas ===")
    try:
        model.fit(X, X, batch_size=cfg.batch_size, epochs=epochs, lr=cfg.learning_rate,
                  print_every=max(250, epochs // 10), epoch_callback=epoch_callback)
    except KeyboardInterrupt:
        print("\n[!] Entrenamiento final abortado. Se evalúa el modelo como quedó.")
    layer.training = False
    model.save(OUTPUT_DIR / "modelo_ganador.pkl", metadata={
        "model_type": "variational_autoencoder", "dataset_path": str(CODE_DIR), "subset_size": len(X),
        "encoder_layers": cfg.encoder_layers, "params": params, "epochs_trained": log[-1][0] if log else 0,
    })
    return model, log


def write_winning_config(params: dict[str, Any], n_samples: int, study: optuna.Study, final: dict[str, Any]) -> Path:
    cfg = config_from_params(params, n_samples, FINAL_EPOCHS, SEED)

    def val(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(val(x) for x in v) + "]"
        return json.dumps(str(v))

    lines = [
        "# Configuración ganadora del estudio ej2 (Optuna)",
        f"# Mejor trial: #{study.best_trial.number} · score {study.best_value:.4f}",
        "", "[dataset]", f"dataset_path = {val('code_dataset/code')}", f"subset_size = {n_samples}",
        "", "[model]", "model_type = \"variational_autoencoder\"",
        f"encoder_layers = {val(cfg.encoder_layers)}", f"decoder_layers = {val(cfg.decoder_layers)}",
        f"hidden_activation = {val(cfg.hidden_activation)}", "latent_activation = \"linear\"",
        "output_activation = \"linear\"", f"residual = {val(cfg.residual)}",
        "kl_loss = \"kl_divergence\"", f"kl_weight = {val(cfg.kl_weight)}",
        "", "[training]", "loss = \"mean_squared_error\"", f"optimizer = {val(cfg.optimizer)}",
        f"learning_rate = {val(cfg.learning_rate)}", f"lr_scheduler = {val(cfg.lr_scheduler)}",
        f"min_lr_ratio = {val(cfg.min_lr_ratio)}", f"decay_rate = {val(cfg.decay_rate)}",
        f"momentum_beta = {val(cfg.momentum_beta)}", f"beta1 = {val(cfg.beta1)}", f"beta2 = {val(cfg.beta2)}",
        f"epsilon = {val(cfg.epsilon)}", f"batch_size = {val(cfg.batch_size)}", f"max_epochs = {FINAL_EPOCHS}",
        f"seed = {SEED}",
        "", "[resultados_finales]",
        *[f'"{k}" = {val(round(v, 6) if isinstance(v, float) else v)}' for k, v in final.items() if not k.startswith("_")],
    ]
    path = OUTPUT_DIR / "configuracion_ganadora.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def final_stage(study: optuna.Study, cache: DatasetCache, compiler: FractalCompilerService, epochs: int) -> list[Path]:
    written: list[Path] = []
    best = study.best_trial
    params = dict(best.params)
    n_samples = dataset_size_for(params["body"], params["latent_dim"])
    X, labels, texts, families = cache.take(n_samples)
    print(f"\n[OK] Mejor trial #{best.number} (score {best.value:.4f}): {params}")

    written.append(write_architecture_diagram(params, n_samples))
    STOP.rearm("Ctrl+C o Shift+Q: corta el entrenamiento final y evalúa el modelo como quedó.")
    model, log = train_final(params, X, epochs)
    written.append(OUTPUT_DIR / "modelo_ganador.pkl")
    if log:
        written.append(write_training_curves(log, params["kl_weight"]))

    print("\n=== Evaluando reconstrucción y generación (compilando programas) ===")
    rng = np.random.default_rng(SEED)
    final = evaluate_model(
        model, X, texts, compiler, rng, n_compile=len(X), n_prior=FINAL_PRIOR_SAMPLES,
        n_local=FINAL_LOCAL_SAMPLES, local_scales=LOCAL_SCALES, keep_samples=True,
    )
    final.update({"n_samples": n_samples, "n_params": parameter_count(params["body"], params["latent_dim"]),
                  "epochs_trained": log[-1][0] if log else 0})
    samples = final.pop("_samples")

    mu, _ = latent_stats(model, X)
    written.append(write_latent_plot(mu, labels, families))
    written.append(write_generation_chart(final))

    # Galería del dataset
    per_family: dict[str, list[int]] = {f: [i for i, fam in enumerate(families) if fam == f][:GALLERY_PER_FAMILY]
                                        for f in FAMILY_COLORS}
    ds_idx = [i for idx in per_family.values() for i in idx]
    ds_compiled = dict(zip(ds_idx, compile_many(compiler, [texts[i] for i in ds_idx])))
    written.append(write_gallery(OUTPUT_DIR / "dataset_muestras.html", f"Dataset: {n_samples} programas", [
        (f"{fam} ({sum(1 for f in families if f == fam)} programas)",
         [_img(ds_compiled[i].get("image_b64"), labels[i]) for i in idx]) for fam, idx in per_family.items()
    ]))

    # Programas nuevos (muestras locales que compilan y no están en el dataset)
    known = set(texts)
    novel: dict[str, tuple[dict[str, Any], int, float]] = {}
    for k in LOCAL_SCALES:
        for text, result, base in samples[f"local_{k}"]:
            if result["ok"] and text not in known and text not in novel:
                novel[text] = (result, base, k)
    gen_dir = OUTPUT_DIR / "programas_generados"
    novel_items: list[str] = []
    originals = compile_many(compiler, [texts[b] for _, b, _ in novel.values()])
    for n, ((text, (result, base, k)), orig) in enumerate(zip(novel.items(), originals), 1):
        stem = f"{n:03d}_variante_de_{labels[base]}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        (gen_dir / f"{stem}.txt").write_text(text + "\n", encoding="utf-8")
        save_bmp(result, gen_dir / f"{stem}.bmp")
        save_bmp(orig, gen_dir / f"{stem}__original.bmp")
        novel_items += [_img(orig.get("image_b64"), f"original: {labels[base]}"),
                        _img(result.get("image_b64"), f"#{n} generado (k = {k})", highlight=True)]
    prior_items = [_img(r.get("image_b64"), f"muestra {i}: " + ("compila" if r["ok"] else "no compila"))
                   for i, (_, r) in enumerate(samples["prior"][:12])]
    prior_code = html.escape(samples["prior"][0][0][:400]) if samples["prior"] else ""

    # Interpolación entre los dos programas más cercanos de la misma familia
    interp_items: list[str] = []
    fam_counts = {f: [i for i, fam in enumerate(families) if fam == f] for f in FAMILY_COLORS}
    fam = max(fam_counts, key=lambda f: len(fam_counts[f]))
    idx = fam_counts[fam]
    if len(idx) >= 2:
        d = np.linalg.norm(mu[idx][:, None] - mu[idx][None], axis=-1)
        np.fill_diagonal(d, np.inf)
        a, b = np.unravel_index(np.argmin(d), d.shape)
        a, b = idx[a], idx[b]
        alphas = np.linspace(0, 1, INTERPOLATION_STEPS)
        with EvalMode(model):
            z = np.array([(1 - t) * mu[a] + t * mu[b] for t in alphas])
            interp_texts = decode_texts(model.decode(z))
        for t, r in zip(alphas, compile_many(compiler, interp_texts)):
            interp_items.append(_img(r.get("image_b64"), f"α = {t:.2f}"))
        interp_title = f"Interpolación {labels[a]} → {labels[b]}"
    else:
        interp_title = "Interpolación (no hay dos programas de la misma familia)"

    written.append(write_gallery(OUTPUT_DIR / "galeria_generacion.html", "VAE como modelo generativo", [
        (f"Programas nuevos que compilan ({len(novel)} distintos) · original vs. generado", novel_items),
        (f"Muestras de N(0, I): {final['prior_valid']:.0%} compilan", prior_items),
        (interp_title, interp_items),
    ], extra_html=f"<h2>Ejemplo de muestra de N(0, I) decodificada</h2><pre>{prior_code}</pre>"))
    final["novel_programs"] = len(novel)
    if novel:
        written.append(gen_dir)

    (OUTPUT_DIR / "metricas_finales.json").write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    written.append(OUTPUT_DIR / "metricas_finales.json")
    written.append(write_winning_config(params, n_samples, study, final))

    print(
        f"\nResultado final: caracteres OK={final['char_acc']:.1%} | compilan={final['rec_compile']:.0%} | "
        f"N(0,I) válidas={final['prior_valid']:.0%} | programas nuevos distintos={len(novel)}"
    )
    return written


# =========================================================================
# MAIN
# =========================================================================
def print_dataset_plan() -> None:
    print("\nTamaño del dataset según la red (latente 8 / 16 / 32):")
    for body in BODIES:
        sizes = " / ".join(str(dataset_size_for(body, z)) for z in LATENT_DIMS)
        print(f"  {body:18s} {parameter_count(body, 16):>10,} parámetros (lat. 16) → {sizes} programas")


def main() -> None:
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Estudio del VAE sobre código de fractales (ej. 2) con Optuna.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Carpeta de resultados.")
    parser.add_argument("--trials", type=int, default=N_TRIALS, help="Cantidad máxima de trials de Optuna.")
    parser.add_argument("--final-epochs", type=int, default=FINAL_EPOCHS, help="Épocas del entrenamiento final.")
    parser.add_argument("--resume", action="store_true", help="Continúa el estudio guardado en optuna.db.")
    parser.add_argument("--skip-final", action="store_true", help="Solo búsqueda y gráficos de Optuna.")
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir if args.output_dir.is_absolute() else (ROOT / args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ensure_code_dataset()
    compiler = FractalCompilerService()
    if not compiler.compiler_path.exists():
        raise SystemExit(f"No se encontró el compilador en {compiler.compiler_path}")
    print_dataset_plan()
    cache = DatasetCache()

    STOP.install()
    written: list[Path] = []
    try:
        study, aborted = run_search(cache, compiler, args.trials, args.resume)
        written += write_optuna_plots(study)
        if not completed_trials(study):
            print("\n[WARN] No terminó ningún trial: no hay configuración ganadora.")
        else:
            written.append(write_exploration_chart(study))
            if aborted:
                print("[INFO] Búsqueda abortada con doble Ctrl+C: no se hace el entrenamiento final.")
            elif not args.skip_final:
                written += final_stage(study, cache, compiler, args.final_epochs)
    finally:
        STOP.uninstall()

    print("\nGráficos y datos:")
    for path in written:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
