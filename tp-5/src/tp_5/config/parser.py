import argparse


def nullable_int(val: str) -> int | None:
    if val.lower() in ("none", "null"):
        return None
    try:
        return int(val)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid int value: '{val}'")


def build_parser() -> argparse.ArgumentParser:
    """Builds and returns the argument parser for Autoencoder, DenoisingAutoencoder & VariationalAutoencoder."""
    parser = argparse.ArgumentParser(
        description="Autoencoder, Denoising Autoencoder & Variational Autoencoder Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to a JSON/TOML configuration file. CLI arguments override JSON/TOML values.",
    )

    # Dataset Parameters
    data = parser.add_argument_group("Dataset Parameters")
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
        help="Optional subset size of patterns to train on (default: full dataset)",
    )

    # Model & Architecture Parameters
    model = parser.add_argument_group("Model & Architecture Parameters")
    model.add_argument(
        "-m",
        "--model-type",
        type=str,
        default="autoencoder",
        choices=["autoencoder", "denoising_autoencoder", "variational_autoencoder"],
        help="Model class name to resolve dynamically via reflection",
    )
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
        help="Activation function class name for hidden layers",
    )
    model.add_argument(
        "--latent-activation",
        type=str,
        default="linear",
        help="Activation function class name for the latent bottleneck layer",
    )
    model.add_argument(
        "--output-activation",
        type=str,
        default="sigmoid",
        help="Activation function class name for the output reconstruction layer",
    )
    model.add_argument("--beta", type=float, default=1.0, help="Beta slope for Sigmoid/Tanh")

    # Denoising & Variational Parameters
    special = parser.add_argument_group("Denoising & Variational Parameters")
    special.add_argument(
        "--noise-type",
        type=str,
        default="bit_flip",
        help="Noise class name in tp_5.engine.noises (e.g. no_noise, bit_flip, gaussian, masking)",
    )
    special.add_argument(
        "--noise-level",
        type=float,
        default=0.1,
        help="Corruption intensity/probability for DenoisingAutoencoder",
    )
    special.add_argument(
        "--eval-noise-levels",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20, 0.30],
        help="Noise levels evaluated in the denoising robustness study",
    )
    special.add_argument(
        "--kl-loss",
        type=str,
        default="kl_divergence",
        help="Divergence loss class name for VariationalAutoencoder",
    )
    special.add_argument(
        "--kl-weight",
        type=float,
        default=0.001,
        help="KL divergence weight (beta) for VariationalAutoencoder",
    )

    # Optimization & Training
    train = parser.add_argument_group("Training & Optimization Parameters")
    train.add_argument("-lr", "--learning-rate", type=float, default=0.005, help="Learning rate")
    train.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="Optimizer class name in tp_5.engine.optimizers (sgd, momentum, rmsprop, adam)",
    )
    train.add_argument("--momentum-beta", type=float, default=0.9, help="Momentum alpha coefficient")
    train.add_argument(
        "--loss",
        type=str,
        default="binary_cross_entropy",
        help="Reconstruction loss class name in tp_5.engine.loss_functions",
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
