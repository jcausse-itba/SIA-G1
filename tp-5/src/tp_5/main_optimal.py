import argparse
from typing import Any

from tp_5.__main__ import build_pipeline
from tp_5.config.validator import build_config_object
from tp_5.main_bayesian import BayesianAutoencoderSearcher, SearchSpaceSpec
from tp_5.training_report import TrainingReporter

# =========================================================================
# HARDCODED OPTIMAL CONFIGURATIONS BY MODEL TYPE
# =========================================================================
OPTIMAL_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "autoencoder": {
        "model_type": "autoencoder",
        "dataset_path": "doc/font.h",
        "encoder_layers": [35, 64, 32, 16, 2],
        "decoder_layers": [16, 32, 64, 35],
        "hidden_activation": "tanh",
        "latent_activation": "linear",
        "output_activation": "sigmoid",
        "loss": "binary_cross_entropy",
        "optimizer": "adam",
        "learning_rate": 0.004,
        "batch_size": 8,
        "max_epochs": 5000,
        "save_model_path": "saved_models/autoencoder_optimal.pkl",
        "output_dir": "results",
    },
    "denoising_autoencoder": {
        "model_type": "denoising_autoencoder",
        "dataset_path": "doc/font.h",
        "encoder_layers": [35, 64, 32, 2],
        "decoder_layers": [32, 64, 35],
        "hidden_activation": "tanh",
        "latent_activation": "tanh",
        "output_activation": "sigmoid",
        "noise_type": "bit_flip",
        "noise_level": 0.10,
        "eval_noise_levels": [0.05, 0.10, 0.20, 0.30],
        "loss": "binary_cross_entropy",
        "optimizer": "adam",
        "learning_rate": 0.003,
        "batch_size": 8,
        "max_epochs": 5000,
        "save_model_path": "saved_models/denoising_autoencoder_optimal.pkl",
        "output_dir": "results",
    },
    "variational_autoencoder": {
        "model_type": "variational_autoencoder",
        "dataset_path": "code_dataset/code",
        "subset_size": 100,
        "encoder_layers": [1000, 256, 128, 16],
        "decoder_layers": [128, 256, 1000],
        "hidden_activation": "tanh",
        "latent_activation": "linear",
        "output_activation": "linear",
        "kl_loss": "kl_divergence",
        "kl_weight": 0.00001,
        "loss": "mean_squared_error",
        "optimizer": "adam",
        "learning_rate": 0.001,
        "batch_size": 16,
        "max_epochs": 2500,
        "save_model_path": "saved_models/vae_code_optimal.pkl",
        "output_dir": "results",
    },
}

# =========================================================================
# BAYESIAN SEARCH SPACES BY MODEL TYPE
# =========================================================================
BAYESIAN_SEARCH_SPECS: dict[str, SearchSpaceSpec] = {
    "autoencoder": SearchSpaceSpec(
        model_type="autoencoder",
        dataset_path="doc/font.h",
        candidate_architectures={
            "35-32-16-2": [35, 32, 16, 2],
            "35-64-32-16-2": [35, 64, 32, 16, 2],
            "35-48-24-2": [35, 48, 24, 2],
        },
        hidden_activations=["tanh", "relu"],
        latent_activations=["linear", "tanh"],
        losses=["binary_cross_entropy", "mean_squared_error"],
        batch_sizes=[4, 8, 16],
        lr_range=(1e-3, 1.5e-2),
        max_trials=30,
        time_per_trial_sec=25.0,
        max_epochs_per_trial=5000,
    ),
    "denoising_autoencoder": SearchSpaceSpec(
        model_type="denoising_autoencoder",
        dataset_path="doc/font.h",
        fixed_overrides={"noise_type": "bit_flip", "noise_level": 0.10},
        max_trials=25,
        time_per_trial_sec=25.0,
        max_epochs_per_trial=5000,
    ),
    "variational_autoencoder": SearchSpaceSpec(
        model_type="variational_autoencoder",
        dataset_path="code_dataset/code",
        candidate_architectures={
            "1000-256-128-8": [1000, 256, 128, 8],
            "1000-256-128-16": [1000, 256, 128, 16],
            "1000-512-256-16": [1000, 512, 256, 16],
        },
        hidden_activations=["tanh", "relu"],
        latent_activations=["linear"],
        output_activation="linear",
        losses=["mean_squared_error"],
        batch_sizes=[16, 32],
        lr_range=(3e-4, 3e-3),
        fixed_overrides={"kl_loss": "kl_divergence", "kl_weight": 1e-5, "subset_size": 100},
        max_trials=15,
        time_per_trial_sec=45.0,
        max_epochs_per_trial=2000,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimal Autoencoder Runner & Bayesian Hyperparameter Optimizer"
    )
    parser.add_argument(
        "-m",
        "--model-type",
        default="variational_autoencoder",
        choices=list(OPTIMAL_MODEL_CONFIGS.keys()),
        help="Model architecture to train or optimize (default: 'variational_autoencoder')",
    )
    parser.add_argument(
        "--bayesian",
        action="store_true",
        help="Run Optuna Bayesian search before generating final reports",
    )
    args = parser.parse_args()

    if args.bayesian:
        spec = BAYESIAN_SEARCH_SPECS[args.model_type]
        searcher = BayesianAutoencoderSearcher(spec=spec, config_factory=build_config_object)
        best_cfg, best_model, best_summary = searcher.run_search()
        dataset = best_cfg.build_dataset()

        reporter = TrainingReporter(output_dir=best_cfg.output_dir, model_name=best_cfg.model_type)
        reporter.write_reports(
            loss_history=[best_summary["loss"]],
            eval_summary=best_summary,
            dataset_inputs=dataset.inputs,
            labels=dataset.labels,
            image_shape=dataset.image_shape,
            model=best_model,
        )
        print(
            f"\n[OK] Best Bayesian Model ({best_cfg.model_type}): "
            f"Arch={best_cfg.encoder_layers}, MaxErr={best_summary['max_pixel_error']}, "
            f"<=1err={best_summary['patterns_le_1_error']}/{best_summary['num_patterns']}"
        )
    else:
        config = build_config_object(OPTIMAL_MODEL_CONFIGS[args.model_type])
        pipeline = build_pipeline(config)
        pipeline.run()


if __name__ == "__main__":
    main()
