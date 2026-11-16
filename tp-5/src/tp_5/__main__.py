from dataclasses import dataclass
from typing import Any

from tp_5.config.loader import load_and_merge_config
from tp_5.config.parser import build_parser
from tp_5.config.validator import AutoencoderConfig, DatasetBundle, build_config_object
from tp_5.engine.models.autoencoder import Autoencoder
from tp_5.training_report import TrainingReporter


@dataclass
class AutoencoderPipeline:
    """Executes training, evaluation, persistence, and reporting via injected dependencies."""

    config: AutoencoderConfig
    dataset: DatasetBundle
    model: Autoencoder
    reporter: TrainingReporter

    def run(self) -> dict[str, Any]:
        print(
            f"\n=== Training {type(self.model).__name__} | "
            f"Patterns={len(self.dataset.inputs)} ({self.dataset.image_shape[0]}x{self.dataset.image_shape[1]}) | "
            f"Encoder={self.config.encoder_layers} ==="
        )

        loss_history = self.model.fit(
            self.dataset.inputs,
            self.dataset.inputs,
            batch_size=self.config.batch_size,
            epochs=self.config.max_epochs,
            lr=self.config.learning_rate,
            print_every=max(1, self.config.max_epochs // 10),
        )

        eval_summary = self.model.evaluate_reconstruction(
            self.dataset.inputs,
            self.dataset.inputs,
            threshold=self.config.threshold,
        )

        if self.config.save_model_path:
            self.model.save(
                self.config.save_model_path,
                metadata={
                    "model_type": self.config.model_type,
                    "dataset_path": str(self.config.dataset_path),
                    "subset_size": self.config.subset_size,
                    "encoder_layers": self.config.encoder_layers,
                    "labels": self.dataset.labels,
                    "max_pixel_error": eval_summary["max_pixel_error"],
                    "mean_pixel_error": eval_summary["mean_pixel_error"],
                },
            )
            print(f"[OK] Model saved to: {self.config.save_model_path}")

        report_paths = self.reporter.write_reports(
            loss_history=loss_history,
            eval_summary=eval_summary,
            dataset_inputs=self.dataset.inputs,
            labels=self.dataset.labels,
            image_shape=self.dataset.image_shape,
            model=self.model,
        )

        if hasattr(self.model, "noise_strategy"):
            report_paths["denoising_study"] = self.reporter.write_denoising_study(
                model=self.model,
                clean_inputs=self.dataset.inputs,
                noise_factory=self.config.build_noise,
                noise_levels=self.config.eval_noise_levels,
            )

        self._print_summary(eval_summary, report_paths)
        return eval_summary

    def _print_summary(self, eval_summary: dict[str, Any], report_paths: dict[str, Any]) -> None:
        print("\n" + "=" * 60)
        print(f" FINAL RESULTS — {type(self.model).__name__.upper()}")
        print("=" * 60)
        print(f" Final Loss                  : {eval_summary['loss']:.6f}")
        print(f" Max Pixel Error (target <=1): {eval_summary['max_pixel_error']} px")
        print(f" Mean Pixel Error            : {eval_summary['mean_pixel_error']:.4f} px")
        print(
            f" Patterns with <= 1 px error : "
            f"{eval_summary['patterns_le_1_error']}/{eval_summary['num_patterns']}"
        )
        print(
            f" Perfect Patterns (0 px err) : "
            f"{eval_summary['perfect_patterns']}/{eval_summary['num_patterns']}"
        )
        print(" Generated Interactive Charts:")
        for name, path in report_paths.items():
            print(f"   - {name}: {path}")
        print("=" * 60)


def build_pipeline(config: AutoencoderConfig) -> AutoencoderPipeline:
    """Wires and injects all collaborators into AutoencoderPipeline."""
    dataset = config.build_dataset()
    model = config.build_model()
    reporter = TrainingReporter(output_dir=config.output_dir, model_name=config.model_type)
    return AutoencoderPipeline(config=config, dataset=dataset, model=model, reporter=reporter)


def main() -> None:
    parser = build_parser()
    try:
        raw_config = load_and_merge_config(parser)
        config = build_config_object(raw_config)
        pipeline = build_pipeline(config)
        pipeline.run()
    except (FileNotFoundError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
