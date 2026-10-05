import os
from typing import Any, Dict

def validate_config(cfg: Dict[str, Any]) -> None:
    """Validates the consistency of the input configuration data for TP3."""
    # Verificación de archivos de dataset
    dataset_path = cfg.get("dataset_path")
    if not dataset_path:
        raise ValueError("A dataset path is required (--dataset-path or dataset_path in the config file).")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")
    
    if cfg.get("test_dataset_path") and not os.path.exists(cfg["test_dataset_path"]):
        raise FileNotFoundError(f"Test dataset file not found at: {cfg['test_dataset_path']}")

    validation_method = cfg.get("validation_method", "split")
    if validation_method == "explicit" and not cfg.get("test_dataset_path"):
        raise ValueError("test_dataset_path is required when validation-method is 'explicit'.")
    if validation_method == "k_fold" and cfg.get("k_folds", 0) < 2:
        raise ValueError("k_folds must be at least 2 for 'k_fold' validation method.")

    if cfg.get("load_model_path") and not os.path.exists(cfg["load_model_path"]):
        raise FileNotFoundError(f"Pre-trained model file not found at: {cfg['load_model_path']}")

    # Validación de hiperparámetros numéricos
    if cfg["learning_rate"] <= 0:
        raise ValueError("Learning rate (--learning-rate) must be positive.")
    
    if cfg["max_epochs"] <= 0:
        raise ValueError("Max epochs (--max-epochs) must be a positive integer.")
        
    if cfg["batch_size"] and cfg["batch_size"] <= 0:
        raise ValueError("Batch size (--batch-size) must be greater than zero.")

    if not (0.0 < cfg["split_ratio"] < 1.0):
        raise ValueError("Split ratio (--split-ratio) must be between 0.0 and 1.0.")

    if not (0.0 <= cfg["threshold"] <= 1.0):
        raise ValueError("Threshold (--threshold) must be between 0.0 and 1.0.")

    if not (0.0 <= cfg.get("fraud_cutoff", 0.5) <= 1.0):
        raise ValueError("Fraud cutoff (--fraud-cutoff) must be between 0.0 and 1.0.")

    if cfg.get("metrics_points", 100) < 1:
        raise ValueError("Metrics points (--metrics-points) must be at least 1.")

    # Validación de la arquitectura
    if cfg["model_type"] == "multilayer":
        if not cfg["architecture"] or len(cfg["architecture"]) < 2:
            raise ValueError("Multilayer perceptron requires at least 2 layer sizes in architecture (Input & Output).")
        if any(size <= 0 for size in cfg["architecture"]):
            raise ValueError("All layer sizes in architecture must be positive integers.")