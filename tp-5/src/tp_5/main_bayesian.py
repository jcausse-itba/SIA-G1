from dataclasses import dataclass, field
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable

import optuna
from optuna.samplers import TPESampler

from tp_5.config.validator import AutoencoderConfig, DatasetBundle, build_config_object


@dataclass
class SearchSpaceSpec:
    """Encapsulates the hyperparameter search space for any Autoencoder architecture."""

    model_type: str = "autoencoder"
    dataset_path: str = "doc/font.h"

    candidate_architectures: dict[str, list[int]] = field(
        default_factory=lambda: {
            "35-32-16-2": [35, 32, 16, 2],
            "35-64-32-16-2": [35, 64, 32, 16, 2],
            "35-48-24-2": [35, 48, 24, 2],
            "35-64-24-2": [35, 64, 24, 2],
        }
    )
    hidden_activations: list[str] = field(default_factory=lambda: ["tanh", "relu", "sigmoid"])
    latent_activations: list[str] = field(default_factory=lambda: ["linear", "tanh"])
    output_activation: str = "sigmoid"
    losses: list[str] = field(default_factory=lambda: ["binary_cross_entropy", "mean_squared_error"])
    optimizers: list[str] = field(default_factory=lambda: ["adam"])
    lr_schedulers: list[str] = field(
        default_factory=lambda: ["cosine_annealing", "exponential_decay", "linear_decay", "constant"]
    )
    batch_sizes: list[int] = field(default_factory=lambda: [4, 8, 16, 32])
    lr_range: tuple[float, float] = (5e-4, 2e-2)
    min_lr_ratio_range: tuple[float, float] = (0.005, 0.1)
    decay_rate_range: tuple[float, float] = (0.01, 0.2)
    beta1_range: tuple[float, float] = (0.85, 0.99)
    beta2_range: tuple[float, float] = (0.95, 0.9999)
    epsilon_range: tuple[float, float] = (1e-8, 1e-6)

    fixed_overrides: dict[str, Any] = field(default_factory=dict)

    max_trials: int = 50
    time_per_trial_sec: float = 30.0
    max_epochs_per_trial: int = 6000
    eval_every_epochs: int = 25


class BayesianAutoencoderSearcher:
    """Generalized Optuna Bayesian Searcher receiving its SearchSpaceSpec and config factory via DI."""

    def __init__(
        self,
        spec: SearchSpaceSpec,
        config_factory: Callable[[dict[str, Any]], AutoencoderConfig] = build_config_object,
        output_dir: str | Path = "optuna_outputs",
    ) -> None:
        self.spec = spec
        self.config_factory = config_factory
        self.output_dir = Path(output_dir) / spec.model_type
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stop_optimization = False
        self.thread_running = False
        self.best_model: Any = None
        self.best_config: AutoencoderConfig | None = None
        self.best_summary: dict[str, Any] | None = None
        self._best_score = float("inf")

    def _listen_for_quit(self) -> None:
        """Background listener catching Shift+Q ('Q') cross-platform."""
        print("\n[INFO] Press 'Shift+Q' at any time to safely stop Bayesian optimization after the current trial.")
        try:
            import msvcrt

            while self.thread_running:
                if msvcrt.kbhit() and msvcrt.getch() == b"Q":
                    self.stop_optimization = True
                    print("\n[!] Shift+Q detected. Stopping optimization at the end of current trial...")
                    break
                time.sleep(0.1)
        except ImportError:
            import select

            try:
                import termios
                import tty

                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                tty.setcbreak(fd)
                has_termios = True
            except Exception:
                has_termios = False

            try:
                while self.thread_running:
                    if has_termios and select.select([sys.stdin], [], [], 0.1)[0]:
                        if sys.stdin.read(1) == "Q":
                            self.stop_optimization = True
                            print("\n[!] Shift+Q detected. Stopping optimization...")
                            break
                    else:
                        time.sleep(0.2)
            finally:
                if has_termios:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _sample_trial_config(self, trial: optuna.Trial) -> AutoencoderConfig:
        arch_key = trial.suggest_categorical("architecture", list(self.spec.candidate_architectures.keys()))
        scheduler_name = trial.suggest_categorical("lr_scheduler", self.spec.lr_schedulers)

        raw: dict[str, Any] = {
            "model_type": self.spec.model_type,
            "dataset_path": self.spec.dataset_path,
            "encoder_layers": self.spec.candidate_architectures[arch_key],
            "hidden_activation": trial.suggest_categorical("hidden_activation", self.spec.hidden_activations),
            "latent_activation": trial.suggest_categorical("latent_activation", self.spec.latent_activations),
            "output_activation": self.spec.output_activation,
            "loss": trial.suggest_categorical("loss", self.spec.losses),
            "optimizer": trial.suggest_categorical("optimizer", self.spec.optimizers),
            "lr_scheduler": scheduler_name,
            "min_lr_ratio": trial.suggest_float("min_lr_ratio", *self.spec.min_lr_ratio_range, log=True),
            "decay_rate": trial.suggest_float("decay_rate", *self.spec.decay_rate_range, log=True),
            "batch_size": trial.suggest_categorical("batch_size", self.spec.batch_sizes),
            "learning_rate": trial.suggest_float("learning_rate", *self.spec.lr_range, log=True),
            "beta1": trial.suggest_float("beta1", *self.spec.beta1_range),
            "beta2": trial.suggest_float("beta2", *self.spec.beta2_range),
            "epsilon": trial.suggest_float("epsilon", *self.spec.epsilon_range, log=True),
            "max_epochs": self.spec.max_epochs_per_trial,
            "seed": 42 + trial.number,
            **self.spec.fixed_overrides,
        }
        return self.config_factory(raw)

    @staticmethod
    def compute_objective_score(summary: dict[str, Any]) -> float:
        max_px = summary["max_pixel_error"]
        mean_px = summary["mean_pixel_error"]
        loss = summary["loss"]
        penalty = max(0, max_px - 1) * 10.0
        return float(penalty + max_px + mean_px + loss)

    def run_search(self, dataset: DatasetBundle | None = None) -> tuple[AutoencoderConfig, Any, dict[str, Any]]:
        if dataset is None:
            first_arch = next(iter(self.spec.candidate_architectures.values()))
            base_cfg = self.config_factory(
                {
                    "model_type": self.spec.model_type,
                    "dataset_path": self.spec.dataset_path,
                    "encoder_layers": first_arch,
                    **self.spec.fixed_overrides,
                }
            )
            dataset = base_cfg.build_dataset()

        study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
        self.thread_running = True
        listener = threading.Thread(target=self._listen_for_quit, daemon=True)
        listener.start()

        def objective(trial: optuna.Trial) -> float:
            cfg = self._sample_trial_config(trial)
            model = cfg.build_model(quantization_scale=dataset.quantization_scale)
            start_time = time.time()
            best_trial_score = float("inf")
            best_trial_summary: dict[str, Any] | None = None

            print(
                f"\n>>> Trial #{trial.number} | Model={cfg.model_type} | Arch={cfg.encoder_layers} | "
                f"Hid={cfg.hidden_activation} | Sched={cfg.lr_scheduler} | "
                f"Batch={cfg.batch_size} | LR={cfg.learning_rate:.5f}"
            )

            def epoch_callback(epoch: int, current_model: Any) -> bool:
                nonlocal best_trial_score, best_trial_summary

                if (time.time() - start_time) >= self.spec.time_per_trial_sec or self.stop_optimization:
                    return True

                if epoch % self.spec.eval_every_epochs == 0:
                    summary = current_model.evaluate_reconstruction(dataset.inputs, dataset.inputs)
                    score = self.compute_objective_score(summary)
                    if score < best_trial_score:
                        best_trial_score = score
                        best_trial_summary = summary

                    if score < self._best_score:
                        self._best_score = score
                        self.best_config = cfg
                        self.best_summary = summary
                        current_model.save(
                            self.output_dir / "best_model.pkl",
                            metadata={
                                "model_type": cfg.model_type,
                                "dataset_path": str(cfg.dataset_path),
                                "subset_size": cfg.subset_size,
                                "encoder_layers": cfg.encoder_layers,
                                "labels": dataset.labels,
                                "max_pixel_error": summary["max_pixel_error"],
                                "mean_pixel_error": summary["mean_pixel_error"],
                            },
                        )

                    trial.report(score, epoch)
                    if summary["max_pixel_error"] == 0 and summary["loss"] < 1e-4:
                        return True

                return False

            model.fit(
                dataset.inputs,
                dataset.inputs,
                batch_size=cfg.batch_size,
                epochs=cfg.max_epochs,
                lr=cfg.learning_rate,
                print_every=max(250, cfg.max_epochs // 4),
                epoch_callback=epoch_callback,
            )

            if best_trial_summary is None:
                best_trial_summary = model.evaluate_reconstruction(dataset.inputs, dataset.inputs)
                best_trial_score = self.compute_objective_score(best_trial_summary)

            print(
                f"    [Trial #{trial.number} Done] MaxErr={best_trial_summary['max_pixel_error']} | "
                f"MeanErr={best_trial_summary['mean_pixel_error']:.2f} | "
                f"<=1err={best_trial_summary['patterns_le_1_error']}/{best_trial_summary['num_patterns']} | "
                f"Loss={best_trial_summary['loss']:.6f}"
            )
            return best_trial_score

        def stop_cb(st: optuna.Study, tr: optuna.trial.FrozenTrial) -> None:
            if self.stop_optimization or (self.best_summary and self.best_summary["max_pixel_error"] == 0):
                st.stop()

        study.optimize(objective, n_trials=self.spec.max_trials, callbacks=[stop_cb], catch=(Exception,))

        self.thread_running = False
        listener.join(timeout=0.5)

        study.trials_dataframe().to_csv(self.output_dir / "optuna_trials.csv", index=False)
        try:
            from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice

            plot_optimization_history(study).write_html(str(self.output_dir / "plot_optimization_history.html"))
            if len(study.trials) > 1:
                plot_param_importances(study).write_html(str(self.output_dir / "plot_param_importances.html"))
            plot_slice(study).write_html(str(self.output_dir / "plot_slice.html"))
        except Exception:
            pass

        if self.best_config is None or self.best_summary is None:
            raise RuntimeError("No valid trial completed during Bayesian search.")

        self.best_config.load_model_path = self.output_dir / "best_model.pkl"
        self.best_model = self.best_config.build_model(quantization_scale=dataset.quantization_scale)
        return self.best_config, self.best_model, self.best_summary
