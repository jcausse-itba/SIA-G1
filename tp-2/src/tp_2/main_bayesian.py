import sys
import time
import threading
from pathlib import Path

import pandas as pd
import optuna
from optuna.samplers import TPESampler

from tp_2.ga.engine import GAEngine
from tp_2.ga.fitness import FitnessEvaluator
from tp_2.image.render import render_to_image
from tp_2.image.utils import ImageUtils

# =========================================================================
# GLOBALS & CONSTANTS
# =========================================================================
IMAGE_PATH = "figures/starry-night.png"
OUTPUT_DIR = Path("optuna_outputs")

MAX_TRIALS         = 100000 
TIME_PER_TRIAL_SEC = 60*2.5       # Limit GA execution to 1 minute per iteration

stop_optimization = False
thread_running = True

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
    """Optuna callback to stop the study if requested by the user."""
    if stop_optimization:
        study.stop()


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function for evaluating GA hyperparameters.
    """
    # 1. Hyperparameter Search Space Definition
    pop_size = trial.suggest_int("pop_size", 20, 1000)
    num_triangles = trial.suggest_int("num_triangles", 30, 100)
    crossover_prob = trial.suggest_float("crossover_prob", 0.1, 1.0)
    mutation_prob = trial.suggest_float("mutation_prob", 0.01, 0.1)
    add_triangle_prob = trial.suggest_float("add_triangle_prob", 0.0, 0.05)
    p_tri = trial.suggest_float("p_tri", 0.1, 0.9)
    p_comp = trial.suggest_float("p_comp", 0.01, 0.5)
    
    # Selection and Mutation Hyperparams
    funsearch_temp = trial.suggest_float("funsearch_temperature", 0.1, 10.0, log=True)
    funsearch_penalty = trial.suggest_float("funsearch_penalty", 0.0, 1.0)
    elitism = trial.suggest_int("elitism", 1, max(2, pop_size // 10))
    
    # Build config dictionary for GAEngine
    cfg = {
        "pop_size": pop_size,
        "num_triangles": num_triangles,
        "children_size": pop_size,
        "crossover_prob": crossover_prob,
        "mutation_prob": mutation_prob,
        "add_triangle_prob": add_triangle_prob,
        "p_tri": p_tri,
        "p_comp": p_comp,
        "elitism": elitism,
        
        # Hardcoded constraints for this experiment
        "parent_selection": "funsearch_priority",
        "survival_selection": "elite",
        "survival_strategy": "additive",
        "crossover": "adaptive_layer_spatial",
        "mutation": "scale_adaptive_gaussian",
        "funsearch_temperature": funsearch_temp,
        "funsearch_penalty": funsearch_penalty,
        
        # System/Time constraints
        "max_generations": 100000, 
        "time_limit_sec": TIME_PER_TRIAL_SEC,
        "save_frames": False,
        "save_interval": 500,
        "output_path": str(OUTPUT_DIR / f"trial_{trial.number}_running.png")
    }

    print(f"\n>>> Iniciando Trial #{trial.number}")
    print(f"    Params: {trial.params}")

    # 2. Prepare Evaluator (Using low res evaluating to squeeze more generations in 60s)
    eval_size = 64
    target_img_eval = ImageUtils.load_target_image(IMAGE_PATH, max_size=(eval_size, eval_size))
    evaluator = FitnessEvaluator(target_img_eval)

    # 3. Run GA Engine
    engine = GAEngine(cfg, evaluator)
    best_individual = engine.run()
    
    final_fitness = best_individual.fitness if best_individual.fitness else 0.0

    # 4. Save Final Renders & History Data
    target_img_full = ImageUtils.load_target_image(IMAGE_PATH)
    full_height, full_width = target_img_full.shape[:2]

    low_res_render = render_to_image(best_individual, eval_size, eval_size)
    high_res_render = render_to_image(best_individual, full_width, full_height)

    ImageUtils.save_image(low_res_render, str(OUTPUT_DIR / f"trial_{trial.number}_low_res.png"))
    ImageUtils.save_image(high_res_render, str(OUTPUT_DIR / f"trial_{trial.number}_high_res.png"))

    history_df = pd.DataFrame(engine.history)
    history_df.to_csv(OUTPUT_DIR / f"trial_{trial.number}_history.csv", index=False)

    return final_fitness


def main():
    global thread_running
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Bayesian Sampler (TPE)
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))

    # Educated Guess Injection
    educated_guess = {
        "pop_size": 50,
        "num_triangles": 50,
        "crossover_prob": 0.85,
        "mutation_prob": 0.20,
        "add_triangle_prob": 0.01,
        "p_tri": 0.30,
        "p_comp": 0.20,
        "funsearch_temperature": 1.0,
        "funsearch_penalty": 0.0,
        "elitism": 2
    }
    study.enqueue_trial(educated_guess)

    print(f"=== Iniciando Optuna Bayesian Search ===")
    print(f" Tiempo límite por iteración: {TIME_PER_TRIAL_SEC}s")

    listener = threading.Thread(target=listen_for_quit, daemon=True)
    listener.start()
    
    # Run optimization
    study.optimize(
        objective, 
        n_trials=MAX_TRIALS, 
        callbacks=[stop_callback],
        catch=(Exception,)
    )

    thread_running = False
    listener.join(timeout=1.0)

    print("\n=========================================")
    print("[OK] Optimización finalizada con éxito.")
    print(f" Mejor Trial    : #{study.best_trial.number}")
    print(f" Mejor Fitness  : {study.best_trial.value:.5f}")
    print(" Mejores Parámetros:")
    for key, value in study.best_trial.params.items():
        print(f"   - {key}: {value}")
    print("=========================================")

    # Export Study Results to CSV
    results_df = study.trials_dataframe()
    results_df.to_csv(OUTPUT_DIR / "optuna_summary_results.csv", index=False)

    # Plotly Visualizations (Saves to interactive HTML)
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice
        
        fig_hist = plot_optimization_history(study)
        fig_hist.write_html(str(OUTPUT_DIR / "plot_optimization_history.html"))
        
        fig_imp = plot_param_importances(study)
        fig_imp.write_html(str(OUTPUT_DIR / "plot_param_importances.html"))
        
        fig_slice = plot_slice(study)
        fig_slice.write_html(str(OUTPUT_DIR / "plot_slice.html"))

        print(f"\n[INFO] Gráficos de Plotly interactivos generados en el directorio: {OUTPUT_DIR}/")
        print("  -> Abre 'plot_optimization_history.html' o 'plot_param_importances.html' en tu navegador.")
        
    except ImportError:
        print("\n[WARN] 'plotly' no está instalado. Ejecuta 'pip install plotly' para generar los gráficos.")


if __name__ == "__main__":
    main()
