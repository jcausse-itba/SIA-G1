import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tp_5.data.header import load_header_dataset
from tp_5.generation.latent import interpolate, latent_grid
from tp_5.generation.noise import apply_noise
from tp_5.models import Autoencoder, DenoisingAutoencoder, VariationalAutoencoder
from tp_5.output.model_payload import model_payload
from tp_5.output.payloads import denoising_payload, generated_payload, reconstruction_payload
from tp_5.output.plots import write_image_grid, write_latent_plot, write_training_plot
from tp_5.output.writer import write_run_artifacts
from tp_5.config.parser import build_parser


TP5_ROOT = Path(__file__).resolve().parents[1]


class DatasetTests(unittest.TestCase):
    def test_font_and_emoji_headers_decode_as_expected_shapes(self) -> None:
        font = load_header_dataset(TP5_ROOT / "data" / "font.h", image_shape=(7, 5))
        emojis = load_header_dataset(TP5_ROOT / "data" / "emojis.h", image_shape=(16, 16))

        self.assertEqual(font.images.shape, (32, 7, 5))
        self.assertEqual(font.labels[1], "a")
        self.assertEqual(font.images[0, 0].tolist(), [0, 0, 1, 0, 0])
        self.assertEqual(emojis.images.shape, (32, 16, 16))
        self.assertEqual(emojis.images.min(), 0.0)
        self.assertEqual(emojis.images.max(), 1.0)

    def test_pattern_selection_rejects_duplicate_and_out_of_range_indices(self) -> None:
        dataset = load_header_dataset(TP5_ROOT / "data" / "font.h", image_shape=(7, 5))
        with self.assertRaises(ValueError):
            dataset.select([1, 1])
        with self.assertRaises(ValueError):
            dataset.select([32])
        with self.assertRaises(ValueError):
            dataset.select(n_patterns=33)


class GenerationTests(unittest.TestCase):
    def test_noise_and_latent_sampling_helpers(self) -> None:
        images = np.zeros((1, 2, 2), dtype=np.float64)
        noisy = apply_noise(images, "bit_flip", 1.0, np.random.default_rng(1))
        self.assertTrue(np.all(noisy == 1.0))

        grid = latent_grid(3, 1.0)
        self.assertEqual(grid.shape, (9, 2))
        points = interpolate(np.array([0.0, 0.0]), np.array([1.0, 2.0]), 3)
        np.testing.assert_allclose(points[1], [0.5, 1.0])

    def test_autoencoder_variants_expose_their_expected_architectures(self) -> None:
        config = vars(build_parser().parse_args([]))
        ae = Autoencoder(35, config)
        dae = DenoisingAutoencoder(35, config)
        vae_config = {**config, "mode": "vae"}
        vae = VariationalAutoencoder(256, vae_config)

        self.assertEqual(ae.encoder_layers, [35, 25, 15, 2])
        self.assertEqual(ae.decoder_layers, [2, 15, 25, 35])
        self.assertEqual(dae.noise_type, "bit_flip")
        self.assertEqual(vae.encoder_layers[-1], 4)
        self.assertEqual(model_payload(ae, "run-1")["latent_dim"], 2)
        with self.assertRaises(NotImplementedError):
            ae.fit(np.zeros((1, 35)), np.zeros((1, 35)))

    def test_reconstruction_and_generation_payload_shapes(self) -> None:
        images = np.zeros((1, 2, 2), dtype=np.float64)
        reconstruction = reconstruction_payload(images, images, ("sample",), 0.5, 1, {0})
        generated = generated_payload(images, np.zeros((1, 2)), 0.5)
        denoising = denoising_payload(
            images,
            {0.1: images[:, None, :, :]},
            {0.1: images[:, None, :, :]},
            0.5,
        )

        self.assertTrue(reconstruction["patterns"][0]["in_training"])
        self.assertEqual(generated["samples"][0]["binarized"], [[0, 0], [0, 0]])
        self.assertEqual(denoising["levels"][0]["samples"][0]["copy_idx"], 0)

    def test_plot_helpers_write_the_requested_formats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_latent_plot(np.array([[0.0, 0.0]]), ["sample"], root / "latent.svg")
            write_image_grid(np.zeros((1, 2, 2)), root / "images.pdf", "Images")
            write_training_plot([{"epoch": 1, "loss": 0.5}], root / "training.png")

            self.assertTrue((root / "latent.svg").is_file())
            self.assertTrue((root / "images.pdf").is_file())
            self.assertTrue((root / "training.png").is_file())

    def test_run_artifacts_write_json_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = write_run_artifacts(
                temporary_directory,
                "run-1",
                {"mode": "ae", "timestamp": "now", "status": "pending", "results": {}},
                {"mode": "ae"},
                [{"epoch": 1, "loss": 0.5}],
                [{"pattern_idx": 0, "z1": 0.1, "z2": 0.2}],
                {"patterns": []},
                {"samples": []},
            )
            run_data = json.loads((run_dir / "run-1_run.json").read_text(encoding="utf-8"))
            self.assertEqual(run_data["mode"], "ae")
            with (run_dir / "run-1_epochs.csv").open(encoding="utf-8", newline="") as file:
                self.assertEqual(next(csv.DictReader(file))["epoch"], "1")
            with (Path(temporary_directory) / "runs.csv").open(encoding="utf-8", newline="") as file:
                self.assertEqual(next(csv.DictReader(file))["run_id"], "run-1")


if __name__ == "__main__":
    unittest.main()
