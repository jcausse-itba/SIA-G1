# main_bayesian.py

import sys
import time
import threading
import json
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

# -- Direct Class Imports (No Reflection/Parser) --
from tp_3.engine.models.mlp import MLP
from tp_3.engine.activation_functions.relu import ReLU
from tp_3.engine.loss_functions.bce import BinaryCrossEntropy
from tp_3.engine.optimizers.adam import Adam

# =========================================================================
# GLOBALS & CONSTANTS
# =========================================================================
DATASET_PATH = Path("./data/more_digits.csv")
OUTPUT_DIR = Path("optuna_outputs")

MAX_TRIALS         = 10000 
TIME_PER_TRIAL_SEC = 60 * 5       # 5 minutes per trial
MAX_EPOCHS         = 100000000    # Effectively no epoch limit

INPUT_SIZE  = 784  # 28x28 images
NUM_CLASSES = 10   # Digit classification

stop_optimization = False
thread_running = True

# =========================================================================
# KEYBOARD LISTENER
# =========================================================================
def listen_for_quit():
    """Background listener catching Shift+Q (capital 'Q') cross-platform."""
    global stop_optimization
    print("\n[INFO] Presiona 'Shift+Q' (Q mayúscula) en cualquier momento para detener la optimización de forma segura.")
    try:
        import msvcrt
        while thread_running:
            if msvcrt.kbhit():
                if msvcrt.getch() == b'Q':
                    stop_optimization = True
                    print("\n[!] Shift+Q detectado. Deteniendo optimización al final del trial actual...")
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
            while thread_running:
                if has_termios:
                    try:
                        if select.select([sys.stdin], [], [], 0.1)[0]:
                            c = sys.stdin.read(1)
                            if c == 'Q':
                                stop_optimization = True
                                print("\n[!] Shift+Q detectado. Deteniendo optimización al final del trial actual...")
                                break
                    except Exception:
                        time.sleep(0.5)
                else:
                    time.sleep(0.5)
        finally:
            if has_termios:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def stop_callback(study, trial):
    if stop_optimization:
        study.stop()

# =========================================================================
# ARCHITECTURE INFERENCE & DATA LOADING
# =========================================================================
def infer_architecture(depth: int) -> list[int]:
    """Infers a well-structured bottleneck architecture exclusively from depth."""
    if depth == 1:
        return [INPUT_SIZE, NUM_CLASSES]
    elif depth == 2:
        return [INPUT_SIZE, 128, NUM_CLASSES]
    elif depth == 3:
        return [INPUT_SIZE, 256, 64, NUM_CLASSES]
    elif depth == 4:
        return [INPUT_SIZE, 512, 128, 32, NUM_CLASSES]
    else:
        # Fallback for depth >= 5
        return [INPUT_SIZE, 512, 256, 128, 64, NUM_CLASSES]


def load_dataset():
    """Loads dataset once into memory to avoid JSON parsing bottlenecks inside the objective loop."""
    print(f"\n[INFO] Loading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)

    # Assumes Col 0 is label, Col 1 is stringified image array
    label_col = df.columns[0]
    image_col = df.columns[1]

    targets = df[label_col].to_numpy().astype(int)
    
    print("[INFO] Parsing string arrays to float matrices (this takes a few seconds)...")
    # Quick parsing
    images = [
        val if isinstance(val, list) else json.loads(str(val))
        for val in df[image_col]
    ]
    X = np.array(images, dtype=np.float64)

    # One-hot encoding for multi-class BCE
    y = np.eye(NUM_CLASSES, dtype=np.float64)[targets]

    # Split Train/Val (80/20) - fixed random seed for consistent evaluation
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split_idx = int(0.8 * len(X))

    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    print(f"[INFO] Dataset ready: Train={len(train_idx)}, Val={len(val_idx)}")
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

# =========================================================================
# OBJECTIVE FUNCTION
# =========================================================================
def objective(trial: optuna.Trial, X_train, y_train, X_val, y_val) -> float:
    # 1. Hyperparameter Search Space
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    
    depth = trial.suggest_int("depth", 1, 5)
    layer_sizes = infer_architecture(depth)
    
    beta1 = trial.suggest_float("beta1", 0.8, 0.999)
    beta2 = trial.suggest_float("beta2", 0.9, 0.9999)
    epsilon = trial.suggest_float("epsilon", 1e-8, 1e-5, log=True)

    print(f"\n>>> Starting Trial #{trial.number} | Depth={depth} | Batch={batch_size} | LR={lr:.5f}")

    # 2. Setup Model instances directly
    act_fn = ReLU()
    out_act_fn = Sigmoid()  # Required to keep predictions [0, 1] for BCE
    loss_fn = BinaryCrossEntropy()
    opt_factory = partial(Adam, beta1=beta1, beta2=beta2, epsilon=epsilon)

    model = MLP(
        layer_sizes=layer_sizes,
        input_activation=act_fn,
        output_activation=out_act_fn,
        loss_function=loss_fn,
        optimizer=opt_factory
    )

    # 3. Time limit logic and callback evaluation
    start_time = time.time()
    best_val_loss = float('inf')

    def epoch_callback(epoch: int, current_model: MLP) -> bool:
        nonlocal best_val_loss

        # Wallclock continuous cut-off check
        elapsed = time.time() - start_time
        if elapsed >= TIME_PER_TRIAL_SEC:
            print(f"    [TimeLimit] Cutoff reached ({elapsed:.1f}s). Stopping Trial.")
            return True 
        
        # Shift+Q flag check
        if stop_optimization:
            return True

        # Perform fast evaluation every 5 epochs
        if epoch % 5 == 0:
            y_pred_val = current_model.forward(X_val)
            # Multiply by batch_size or average natively? Assuming BCE returns scalar mean
            val_loss = float(loss_fn.compute(y_pred_val, y_val))
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            # Report for continuous optimization logic / pruning
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return False

    # 4. Train Model
    try:
        model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=MAX_EPOCHS,  # Enforced no epoch limit, constrained by time
            lr=lr,
            print_every=10, 
            epoch_callback=epoch_callback
        )
    except optuna.TrialPruned:
        print(f"    [Pruned] Trial #{trial.number} pruned by Optuna.")
        raise
    except Exception as e:
        print(f"[ERROR] Trial #{trial.number} failed with error: {e}")
        return float('inf')

    return best_val_loss

# =========================================================================
# MAIN
# =========================================================================
def main():
    global thread_running
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset fully to RAM initially so it isn't bottlenecking trials
    X_train, y_train, X_val, y_val = load_dataset()

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))

    print(f"\n=== Iniciando Optuna Bayesian Search ===")
    print(f" Límite de tiempo por trial: {TIME_PER_TRIAL_SEC}s")
    
    listener = threading.Thread(target=listen_for_quit, daemon=True)
    listener.start()

    # Create partial objective locked to our dataset variables
    bound_objective = partial(
        objective, 
        X_train=X_train, y_train=y_train, 
        X_val=X_val, y_val=y_val
    )

    study.optimize(
        bound_objective, 
        n_trials=MAX_TRIALS, 
        callbacks=[stop_callback],
        catch=(Exception,)
    )

    thread_running = False
    listener.join(timeout=1.0)

    # =========================================================================
    # POST-PROCESSING
    # =========================================================================
    print("\n=========================================")
    print("[OK] Optimización finalizada con éxito.")
    
    if len(study.trials) > 0 and study.best_trial:
        print(f" Mejor Trial    : #{study.best_trial.number}")
        print(f" Mejor Val Loss : {study.best_trial.value:.5f}")
        print(" Mejores Parámetros:")
        for key, value in study.best_trial.params.items():
            print(f"   - {key}: {value}")
    print("=========================================")

    # Continuous save triggers
    results_df = study.trials_dataframe()
    results_df.to_csv(OUTPUT_DIR / "optuna_summary_results.csv", index=False)

    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice

        fig_hist = plot_optimization_history(study)
        fig_hist.write_html(str(OUTPUT_DIR / "plot_optimization_history.html"))

        fig_imp = plot_param_importances(study)
        fig_imp.write_html(str(OUTPUT_DIR / "plot_param_importances.html"))

        fig_slice = plot_slice(study)
        fig_slice.write_html(str(OUTPUT_DIR / "plot_slice.html"))

        print(f"\n[INFO] Gráficos de Plotly interactivos generados en el directorio: {OUTPUT_DIR}/")
    except ImportError:
        print("\n[WARN] 'plotly' no está instalado. Ejecuta 'pip install plotly' para generar los gráficos.")


if __name__ == "__main__":
    main()
