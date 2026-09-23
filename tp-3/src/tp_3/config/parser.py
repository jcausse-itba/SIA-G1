import argparse

def build_parser() -> argparse.ArgumentParser:
    """Builds and returns the argument parser for TP3: Perceptron & Multilayer Perceptron."""
    parser = argparse.ArgumentParser(
        description="TP3: Simple and Multilayer Perceptron",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-c", "--config", type=str, default=None,
                        help="Path to a JSON/TOML configuration file. CLI arguments override JSON/TOML values.")

    # Data Parameters
    data = parser.add_argument_group("Dataset Parameters")
    data.add_argument("-d", "--dataset-path", type=str, required=True, help="Path to the training dataset CSV")
    data.add_argument("--test-dataset-path", type=str, default=None, help="Path to the testing dataset CSV (optional)")

    # Model Parameters
    model = parser.add_argument_group("Model Architecture")
    model.add_argument("-m", "--model-type", type=str, default="multilayer",
                       choices=["simple_linear", "simple_non_linear", "multilayer"],
                       help="Type of perceptron model")
    model.add_argument("-a", "--architecture", type=int, nargs="+", default=[2, 3, 1],
                       help="Layer structure for Multilayer Perceptron (e.g. 2 3 1)")
    model.add_argument("--activation", type=str, default="sigmoid",
                       choices=["step", "linear", "sigmoid", "tanh", "relu"],
                       help="Activation function")
    model.add_argument("--beta", type=float, default=1.0, help="Beta coefficient for non-linear activations")

    # Optimization & Training
    train = parser.add_argument_group("Training & Optimization Parameters")
    train.add_argument("-lr", "--learning-rate", type=float, default=0.01, help="Learning rate (eta)")
    train.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "momentum", "adam"],
                       help="Optimization algorithm")
    train.add_argument("--momentum-beta", type=float, default=0.9, help="Beta coefficient for Momentum")
    train.add_argument("--batch-size", type=int, default=32, help="Batch size for training")

    # Stopping Conditions
    stop = parser.add_argument_group("Stopping Conditions")
    stop.add_argument("--max-epochs", type=int, default=1000, help="Maximum number of epochs")
    stop.add_argument("--target-error", type=float, default=1e-4, help="Target minimum error to stop training")
    stop.add_argument("--target-accuracy", type=float, default=0.98, help="Target accuracy to stop training")

    # Generalization & Validation
    gen = parser.add_argument_group("Generalization Parameters")
    gen.add_argument("--split-ratio", type=float, default=0.8, help="Train/Validation split ratio")
    gen.add_argument("--k-folds", type=int, default=5, help="Number of folds for Cross Validation")
    gen.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for classification")

    # Persistence
    pers = parser.add_argument_group("Model Persistence")
    pers.add_argument("--save-model-path", type=str, default=None, help="Path to save trained weights/model")
    pers.add_argument("--load-model-path", type=str, default=None, help="Path to load pre-trained weights")

    return parser