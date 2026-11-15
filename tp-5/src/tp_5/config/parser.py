import argparse


def nullable_int(val: str) -> int | None:
    if val.lower() in ("none", "null"):
        return None
    try:
        return int(val)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid int value: '{val}'")


def build_parser() -> argparse.ArgumentParser:
    """Builds and returns the argument parser for TP5: Deep Learning (Autoencoders & VAE)."""
    parser = argparse.ArgumentParser(
        description="TP5: Basic Autoencoder, Denoising Autoencoder & Variational Autoencoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to a JSON/TOML configuration file. CLI arguments override JSON/TOML values.",
    )

    # Exercise & Dataset Parameters
    data = parser.add_argument_group("Exercise & Dataset Parameters")
    data.add_argument(
        "-e",
        "--exercise",
        type=str,
        default="1a",
        choices=["1a", "1b", "2"],
        help="Assignment exercise to run: '1a' (Basic AE), '1b' (Denoising AE), or '2' (VAE)",
    )
    data.add_argument(
        "-d",
        "--dataset-path",
        type=str,
        default="doc/font.h",
        help="Path to font.h or CSV dataset",
    )
    data.add_argument(
        "--subset-size",
        type=nullable_int,
        default=None,
        help="Optional subset size of characters to train on (default: all 32 patterns)",
    )

    # Architecture Parameters
    model = parser.add_argument_group("Autoencoder Architecture")
    model.add_argument(
        "-a",
        "--encoder-layers",
        type=int,
        nargs="+",
        default=[35, 32, 16, 2],
        help="Encoder layer dimensions ending with latent dim (decoder mirrors automatically if omitted)",
    )
    model.add_argument(
        "--decoder-layers",
        type=int,
        nargs="+",
        default=None,
        help="Explicit decoder layer dimensions after the latent layer (e.g. 16 32 35)",
    )
    model.add_argument(
        "--hidden-activation",
        type=str,
        default="tanh",
        choices=["linear", "sigmoid", "tanh", "relu"],
        help="Activation function for encoder/decoder hidden layers",
    )
    model.add_argument(
        "--latent-activation",
        type=str,
        default="linear",
        choices=["linear", "sigmoid", "tanh", "relu"],
        help="Activation function for the 2D latent bottleneck layer",
    )
    model.add_argument(
        "--output-activation",
        type=str,
        default="sigmoid",
        choices=["linear", "sigmoid", "tanh", "relu"],
        help="Activation function for the reconstruction output layer",
    )
    model.add_argument("--beta", type=float, default=1.0, help="Beta slope for Sigmoid/Tanh")

    # Denoising & VAE Specifics
    special = parser.add_argument_group("Denoising (1b) & VAE (2) Parameters")
    special.add_argument(
        "--noise-type",
        type=str,
        default="bit_flip",
        choices=["none", "bit_flip", "gaussian", "masking"],
        help="Noise corruption strategy for Exercise 1b",
    )
    special.add_argument(
        "--noise-level",
        type=float,
        default=0.1,
        help="Corruption intensity/probability for Exercise 1b",
    )
    special.add_argument(
        "--eval-noise-levels",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20, 0.30],
        help="Noise levels evaluated in the Exercise 1b robustness study",
    )
    special.add_argument(
        "--kl-weight",
        type=float,
        default=0.001,
        help="KL divergence weight (beta) for Variational Autoencoder in Exercise 2",
    )

    # Optimization & Training
    train = parser.add_argument_group("Training & Optimization Parameters")
    train.add_argument("-lr", "--learning-rate", type=float, default=0.005, help="Learning rate")
    train.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["sgd", "momentum", "rmsprop", "adam"],
        help="Optimization algorithm",
    )
    train.add_argument("--momentum-beta", type=float, default=0.9, help="Momentum alpha coefficient")
    train.add_argument(
        "--loss",
        type=str,
        default="binary_cross_entropy",
        choices=["mean_squared_error", "binary_cross_entropy"],
        help="Reconstruction loss function",
    )
    train.add_argument("--batch-size", type=nullable_int, default=8, help="Mini-batch size")
    train.add_argument("--max-epochs", type=int, default=4000, help="Maximum training epochs")
    train.add_argument("--threshold", type=float, default=0.5, help="Pixel binarization threshold")
    train.add_argument("--seed", type=int, default=42, help="Random seed")

    # Persistence & Output
    pers = parser.add_argument_group("Persistence & Reports")
    pers.add_argument("--save-model-path", type=str, default=None, help="Path to save trained model (.pkl)")
    pers.add_argument("--load-model-path", type=str, default=None, help="Path to load pre-trained model (.pkl)")
    pers.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory where interactive HTML reports are saved",
    )

    return parser
