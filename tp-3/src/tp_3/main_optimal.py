# main_optimal.py

import json
import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor

from tp_3.engine.models.mlp import MLP
from tp_3.engine.activation_functions.relu import ReLU
from tp_3.engine.activation_functions.sigmoid import Sigmoid 
from tp_3.engine.loss_functions.bce import BinaryCrossEntropy
from tp_3.engine.optimizers.adam import Adam

# =========================================================================
# GLOBALS & CONSTANTS (OPTIMAL HYPERPARAMETERS)
# =========================================================================
DATASET_PATH = Path("./data/more_digits.csv")
MODELS_DIR = Path("saved_models")

# Training loops & concurrency
NUM_MODELS_TO_TRAIN = 8
EPOCHS = 150 
NUM_PROCESSES = 4 

# Optimal Hyperparameters from Optuna
BATCH_SIZE = 128
LR = 0.0019381547751445255
BETA1 = 0.8058612186367159
BETA2 = 0.9325569828221307
EPSILON = 3.726638258124091e-08

# Depth 5 inferred architecture
ARCHITECTURE = [784, 512, 256, 128, 64, 10]
NUM_CLASSES = 10

# =========================================================================
# DATA LOADING
# =========================================================================
def load_dataset():
    """Loads and preprocesses the dataset."""
    print(f"\n[INFO] Loading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)

    label_col = df.columns[0]
    image_col = df.columns[1]

    targets = df[label_col].to_numpy().astype(int)
    
    print("[INFO] Parsing string arrays to float matrices...")
    images = [
        val if isinstance(val, list) else json.loads(str(val))
        for val in df[image_col]
    ]
    X = np.array(images, dtype=np.float64)

    y = np.eye(NUM_CLASSES, dtype=np.float64)[targets]

    # Split Train/Val (80/20) - fixed random seed for consistent evaluation
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split_idx = int(0.8 * len(X))

    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    print(f"[INFO] Dataset ready: Train={len(train_idx)}, Val={len(val_idx)}")
    
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

# =========================================================================
# METRICS
# =========================================================================
def calculate_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculates accuracy by taking the argmax of one-hot/probability arrays."""
    pred_classes = np.argmax(y_pred, axis=1)
    true_classes = np.argmax(y_true, axis=1)
    return np.mean(pred_classes == true_classes)

# =========================================================================
# WORKER TASK
# =========================================================================
def train_single_model(i: int, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
    """Trains a single MLP instance in its own process."""
    act_fn = ReLU()
    out_act_fn = Sigmoid()  
    loss_fn = BinaryCrossEntropy()
    opt_factory = partial(Adam, beta1=BETA1, beta2=BETA2, epsilon=EPSILON)

    print(f"\n--- Entrenando Modelo {i}/{NUM_MODELS_TO_TRAIN} ---")
    
    # Instantiate a fresh model
    model = MLP(
        layer_sizes=ARCHITECTURE,
        input_activation=act_fn,
        output_activation=out_act_fn,
        loss_function=loss_fn,
        optimizer=opt_factory
    )
    
    # Train
    model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LR,
        print_every=10
    )
    
    # Evaluate validation set
    y_pred_val = model.forward(X_val)
    val_loss = float(loss_fn.compute(y_pred_val, y_val))
    val_acc = calculate_accuracy(y_pred_val, y_val)
    
    print(f"-> Modelo {i} finalizado | Val Loss: {val_loss:.5f} | Val Accuracy: {val_acc:.5f}")
    
    # Modularized saving using BaseModel.save()
    filename = f"mlp_model_{i}_loss_{val_loss:.4f}_acc_{val_acc:.4f}.pkl"
    filepath = MODELS_DIR / filename
    model.save(filepath, metadata={"val_loss": val_loss, "val_acc": val_acc})
        
    print(f"[OK] Modelo guardado en: {filepath}")

# =========================================================================
# MAIN LOOP
# =========================================================================
def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    X_train, y_train, X_val, y_val = load_dataset()

    print(f"\n=== Iniciando Generación de Modelos ({NUM_PROCESSES} procesos en paralelo) ===")
    
    train_task = partial(
        train_single_model, 
        X_train=X_train, y_train=y_train, 
        X_val=X_val, y_val=y_val
    )

    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        list(executor.map(train_task, range(1, NUM_MODELS_TO_TRAIN + 1)))

    print(f"\n[INFO] {NUM_MODELS_TO_TRAIN} modelos generados exitosamente en la carpeta '{MODELS_DIR}'.")

if __name__ == "__main__":
    main()
