"""
Module to handle letters task (2.2) for Hopfield networks.

Exposes a single entry-point `run_letters_task()` which is used to initiate the task.
"""

__all__ = ["LETTERS_5X5", "get_letter_patterns", "run_letters_task"]

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .hopfield import HopfieldNetwork, apply_noise, hamming_distance
from .plots import (
    BINARY_SCALE,
    DIVERGING,
    format_ascii_grid,
    out_dir,
    save_csv,
    save_fig,
)

LETTER_SHAPE = (5, 5)
N_PIXELS = 25
THEORETICAL_CAPACITY = 0.138 * N_PIXELS  # ~3.45 patrones para N=25

# Diccionario completo A-Z en grillas de 5x5 con {-1, +1}.
# La letra "J" respeta exactamente la matriz de ejemplo de la página 6 del enunciado.
LETTERS_5X5: Dict[str, np.ndarray] = {
    "A": np.array([
        [-1, -1,  1, -1, -1],
        [-1,  1, -1,  1, -1],
        [ 1,  1,  1,  1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
    ], dtype=np.int8),
    "B": np.array([
        [ 1,  1,  1,  1, -1],
        [ 1, -1, -1, -1,  1],
        [ 1,  1,  1,  1, -1],
        [ 1, -1, -1, -1,  1],
        [ 1,  1,  1,  1, -1],
    ], dtype=np.int8),
    "C": np.array([
        [-1,  1,  1,  1,  1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [-1,  1,  1,  1,  1],
    ], dtype=np.int8),
    "D": np.array([
        [ 1,  1,  1,  1, -1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1,  1,  1,  1, -1],
    ], dtype=np.int8),
    "E": np.array([
        [ 1,  1,  1,  1,  1],
        [ 1, -1, -1, -1, -1],
        [ 1,  1,  1,  1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1,  1,  1,  1,  1],
    ], dtype=np.int8),
    "F": np.array([
        [ 1,  1,  1,  1,  1],
        [ 1, -1, -1, -1, -1],
        [ 1,  1,  1,  1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
    ], dtype=np.int8),
    "G": np.array([
        [-1,  1,  1,  1,  1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1,  1,  1,  1],
        [ 1, -1, -1, -1,  1],
        [-1,  1,  1,  1,  1],
    ], dtype=np.int8),
    "H": np.array([
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1,  1,  1,  1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
    ], dtype=np.int8),
    "I": np.array([
        [ 1,  1,  1,  1,  1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
        [ 1,  1,  1,  1,  1],
    ], dtype=np.int8),
    "J": np.array([
        [ 1,  1,  1,  1,  1],
        [-1, -1, -1,  1, -1],
        [-1, -1, -1,  1, -1],
        [ 1, -1, -1,  1, -1],
        [ 1,  1,  1, -1, -1],
    ], dtype=np.int8),
    "K": np.array([
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1,  1, -1],
        [ 1,  1,  1, -1, -1],
        [ 1, -1, -1,  1, -1],
        [ 1, -1, -1, -1,  1],
    ], dtype=np.int8),
    "L": np.array([
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1,  1,  1,  1, -1],
    ], dtype=np.int8),
    "M": np.array([
        [ 1, -1, -1, -1,  1],
        [ 1,  1, -1,  1,  1],
        [ 1, -1,  1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
    ], dtype=np.int8),
    "N": np.array([
        [ 1, -1, -1, -1,  1],
        [ 1,  1, -1, -1,  1],
        [ 1, -1,  1, -1,  1],
        [ 1, -1, -1,  1,  1],
        [ 1, -1, -1, -1,  1],
    ], dtype=np.int8),
    "O": np.array([
        [-1,  1,  1,  1, -1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [-1,  1,  1,  1, -1],
    ], dtype=np.int8),
    "P": np.array([
        [ 1,  1,  1,  1, -1],
        [ 1, -1, -1, -1,  1],
        [ 1,  1,  1,  1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
    ], dtype=np.int8),
    "Q": np.array([
        [-1,  1,  1,  1, -1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1,  1, -1,  1],
        [ 1, -1, -1,  1, -1],
        [-1,  1,  1, -1,  1],
    ], dtype=np.int8),
    "R": np.array([
        [ 1,  1,  1,  1, -1],
        [ 1, -1, -1, -1,  1],
        [ 1,  1,  1,  1, -1],
        [ 1, -1, -1,  1, -1],
        [ 1, -1, -1, -1,  1],
    ], dtype=np.int8),
    "S": np.array([
        [-1,  1,  1,  1,  1],
        [ 1, -1, -1, -1, -1],
        [-1,  1,  1,  1, -1],
        [-1, -1, -1, -1,  1],
        [ 1,  1,  1,  1, -1],
    ], dtype=np.int8),
    "T": np.array([
        [ 1,  1,  1,  1,  1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
    ], dtype=np.int8),
    "U": np.array([
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [-1,  1,  1,  1, -1],
    ], dtype=np.int8),
    "V": np.array([
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [-1,  1, -1,  1, -1],
        [-1, -1,  1, -1, -1],
    ], dtype=np.int8),
    "W": np.array([
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1,  1, -1,  1],
        [ 1,  1, -1,  1,  1],
        [ 1, -1, -1, -1,  1],
    ], dtype=np.int8),
    "X": np.array([
        [ 1, -1, -1, -1,  1],
        [-1,  1, -1,  1, -1],
        [-1, -1,  1, -1, -1],
        [-1,  1, -1,  1, -1],
        [ 1, -1, -1, -1,  1],
    ], dtype=np.int8),
    "Y": np.array([
        [ 1, -1, -1, -1,  1],
        [-1,  1, -1,  1, -1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
    ], dtype=np.int8),
    "Z": np.array([
        [ 1,  1,  1,  1,  1],
        [-1, -1, -1,  1, -1],
        [-1, -1,  1, -1, -1],
        [-1,  1, -1, -1, -1],
        [ 1,  1,  1,  1,  1],
    ], dtype=np.int8),
}


def get_letter_patterns(letters: Sequence[str]) -> Tuple[List[str], np.ndarray]:
    """Devuelve la lista de nombres en mayúscula y la matriz (P, 25) de patrones en {-1, +1}."""
    names = [ch.upper() for ch in letters]
    for name in names:
        if name not in LETTERS_5X5:
            raise ValueError(f"Letra {name!r} no disponible en LETTERS_5X5.")
    patterns = np.stack([LETTERS_5X5[name].ravel() for name in names], axis=0)
    return names, patterns


def _add_pattern_heatmap(
    fig: go.Figure,
    vec: np.ndarray,
    row: int,
    col: int,
    diff_ref: np.ndarray | None = None,
) -> None:
    """Agrega una grilla 5x5 a un subplot de Plotly."""
    mat = vec.reshape(LETTER_SHAPE)[::-1]  # invertimos eje Y visual para que fila 0 quede arriba
    text = [["+1" if v > 0 else "-1" for v in r] for r in mat]
    if diff_ref is not None:
        ref = diff_ref.reshape(LETTER_SHAPE)[::-1]
        # Codificamos: -1 = blanco (-1), +1 = azul oscuro (+1), error/diferencia = rojo
        z = np.where(mat != ref, 0.0, mat.astype(float))
        colorscale = [[0.0, "#f5f6fa"], [0.5, "#e74c3c"], [1.0, "#2c3e50"]]
    else:
        z = mat.astype(float)
        colorscale = BINARY_SCALE

    fig.add_heatmap(
        z=z,
        zmin=-1,
        zmax=1,
        colorscale=colorscale,
        showscale=False,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 10},
        xgap=2,
        ygap=2,
        row=row,
        col=col,
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=row, col=col)
    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        scaleanchor=f"x{(row - 1) * fig._grid_ref[0].__len__() + col if fig._grid_ref else ''}",
        row=row,
        col=col,
    )


def plot_stored_patterns(net: HopfieldNetwork, out: str) -> pd.DataFrame:
    """Grafica los patrones 5x5 almacenados y guarda tabla de estabilidad/energía."""
    p = net.n_patterns
    rows_info = []
    titles = []
    for idx, name in enumerate(net.names):
        pat = net.patterns[idx]
        e = net.energy(pat)
        stable = net.is_fixed_point(pat)
        rows_info.append({
            "letter": name,
            "energy": round(e, 4),
            "is_fixed_point": stable,
            "active_pixels_pos": int((pat == 1).sum()),
            "active_pixels_neg": int((pat == -1).sum()),
        })
        state_lbl = "estable" if stable else "INESTABLE"
        titles.append(f"Letra {name}<br>E = {e:.2f} ({state_lbl})")

    df_stored = pd.DataFrame(rows_info)
    save_csv(df_stored, out, "patrones_almacenados")

    fig = make_subplots(
        rows=1,
        cols=p,
        subplot_titles=titles,
        horizontal_spacing=0.06,
    )
    for idx in range(p):
        _add_pattern_heatmap(fig, net.patterns[idx], row=1, col=idx + 1)

    fig.update_layout(
        title="Patrones de letras (5×5) almacenados en la red de Hopfield",
        height=340,
        width=max(260 * p, 700),
    )
    save_fig(fig, out, "patrones_almacenados")
    return df_stored


def analyze_orthogonality(net: HopfieldNetwork, out: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula y grafica la matriz de ortogonalidad (solapamiento y distancia de Hamming)."""
    overlap = net.orthogonality_matrix()
    dots = net.dot_product_matrix()
    p = net.n_patterns

    # Tabla matricial de solapamientos
    df_mat = pd.DataFrame(overlap.round(4), index=net.names, columns=net.names)
    save_csv(df_mat, out, "ortogonalidad_matriz", index=True)

    # Tabla de pares (mu < nu)
    pair_rows = []
    for i in range(p):
        for j in range(i + 1, p):
            dot_ij = int(dots[i, j])
            ov_ij = float(overlap[i, j])
            ham_ij = hamming_distance(net.patterns[i], net.patterns[j])
            angle_deg = float(np.degrees(np.arccos(np.clip(ov_ij, -1.0, 1.0))))
            pair_rows.append({
                "pair": f"{net.names[i]}-{net.names[j]}",
                "letter_1": net.names[i],
                "letter_2": net.names[j],
                "dot_product": dot_ij,
                "normalized_overlap": round(ov_ij, 4),
                "abs_overlap": round(abs(ov_ij), 4),
                "hamming_distance": ham_ij,
                "angle_deg": round(angle_deg, 2),
            })
    df_pairs = pd.DataFrame(pair_rows).sort_values("abs_overlap", ascending=False)
    save_csv(df_pairs, out, "ortogonalidad_pares")

    # Heatmap interactivo con <xi^mu, xi^nu> y solapamiento m_{mu,nu}
    text_annot = [
        [f"m={overlap[i, j]:+.2f}<br>⟨ξ,ξ⟩={int(dots[i, j])}" for j in range(p)]
        for i in range(p)
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=overlap,
            x=net.names,
            y=net.names,
            zmin=-1.0,
            zmax=1.0,
            colorscale=DIVERGING,
            text=text_annot,
            texttemplate="%{text}",
            colorbar={"title": "Solapamiento m"},
        )
    )
    off_diag = np.abs(overlap[~np.eye(p, dtype=bool)])
    mean_abs = float(off_diag.mean()) if off_diag.size else 0.0
    max_abs = float(off_diag.max()) if off_diag.size else 0.0
    fig.update_layout(
        title=(
            f"Matriz de Ortogonalidad / Solapamiento entre letras "
            f"(|m| medio = {mean_abs:.3f}, |m| máx = {max_abs:.3f})"
        ),
        xaxis_title="Letra",
        yaxis_title="Letra",
        yaxis_autorange="reversed",
        height=500,
        width=580,
    )
    save_fig(fig, out, "ortogonalidad")
    return df_mat, df_pairs


def _greedy_orthogonal_order(seed_letters: Sequence[str], max_p: int = 10) -> List[str]:
    """Extiende seed_letters eligiendo sucesivamente la letra A-Z de menor solapamiento máximo."""
    chosen = [ch.upper() for ch in seed_letters]
    remaining = [ch for ch in LETTERS_5X5 if ch not in chosen]
    while len(chosen) < max_p and remaining:
        best_ch = min(
            remaining,
            key=lambda c: max(
                abs(int(LETTERS_5X5[c].ravel() @ LETTERS_5X5[k].ravel())) / N_PIXELS
                for k in chosen
            ),
        )
        chosen.append(best_ch)
        remaining.remove(best_ch)
    return chosen[:max_p]


def analyze_capacity(cfg: Dict[str, Any], out: str) -> pd.DataFrame:
    """Analiza la capacidad de almacenamiento de la red (P = 1..10) comparando baja vs alta correlación."""
    rng = np.random.default_rng(cfg["seed"])
    max_p = 10
    ortho_seq = _greedy_orthogonal_order(cfg["letters"], max_p=max_p)
    correlated_seq = ["E", "F", "P", "B", "R", "C", "G", "S", "H", "K"][:max_p]

    rows = []
    for group_label, seq in (
        ("Configuradas + baja correlación", ortho_seq),
        ("Letras parecidas (alta correlación)", correlated_seq),
    ):
        for p in range(1, max_p + 1):
            sub_names = seq[:p]
            _, sub_pats = get_letter_patterns(sub_names)
            sub_net = HopfieldNetwork(sub_pats, sub_names)

            ov = sub_net.orthogonality_matrix()
            off = np.abs(ov[~np.eye(p, dtype=bool)])
            mean_abs_ov = float(off.mean()) if off.size else 0.0

            # 1) Estabilidad sin ruido (puntos fijos exactos)
            n_stable = sum(1 for pat in sub_pats if sub_net.is_fixed_point(pat))
            stable_ratio = n_stable / p

            # 2) Recuperación con ruido leve (p=0.12 -> 3 píxeles invertidos)
            n_trials = 15
            hits_noise = 0
            for idx, pat in enumerate(sub_pats):
                for _ in range(n_trials):
                    q = apply_noise(pat, 0.12, rng, noise_type="white", shape=LETTER_SHAPE)
                    res = sub_net.predict(
                        q,
                        update_mode=cfg["update_mode"],
                        max_iterations=cfg["max_iterations"],
                        rng=rng,
                    )
                    outcome, _, _ = sub_net.classify_state(res.final_state, res.converged, target_idx=idx)
                    if outcome == "recovered":
                        hits_noise += 1
            recovery_noise_12 = hits_noise / (p * n_trials)

            rows.append({
                "sequence_type": group_label,
                "n_patterns": p,
                "load_alpha": round(p / N_PIXELS, 4),
                "letters": ",".join(sub_names),
                "mean_abs_overlap": round(mean_abs_ov, 4),
                "stable_patterns": n_stable,
                "stable_ratio": round(stable_ratio, 4),
                "recovery_rate_noise_12pct": round(recovery_noise_12, 4),
            })

    df_cap = pd.DataFrame(rows)
    save_csv(df_cap, out, "capacidad")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Fracción de patrones estables (sin ruido)",
            "Tasa de recuperación con ruido (12% = 3 píxeles)",
        ),
    )
    colors = {"Configuradas + baja correlación": "#1e8449", "Letras parecidas (alta correlación)": "#c0392b"}

    for label, grp in df_cap.groupby("sequence_type", sort=False):
        c = colors[label]
        fig.add_scatter(
            x=grp["n_patterns"],
            y=grp["stable_ratio"],
            mode="lines+markers",
            name=label,
            legendgroup=label,
            line={"color": c, "width": 2.5},
            hovertext=grp["letters"],
            row=1,
            col=1,
        )
        fig.add_scatter(
            x=grp["n_patterns"],
            y=grp["recovery_rate_noise_12pct"],
            mode="lines+markers",
            name=label,
            legendgroup=label,
            showlegend=False,
            line={"color": c, "width": 2.5, "dash": "solid"},
            hovertext=grp["letters"],
            row=1,
            col=2,
        )

    for col in (1, 2):
        fig.add_vline(
            x=THEORETICAL_CAPACITY,
            line_dash="dash",
            line_color="#555555",
            annotation_text=f"C ≈ 0.138·N = {THEORETICAL_CAPACITY:.2f}",
            annotation_position="top right",
            row=1,
            col=col,
        )
        fig.update_xaxes(title_text="Cantidad de patrones almacenados (P)", dtick=1, row=1, col=col)
        fig.update_yaxes(range=[-0.05, 1.05], title_text="Proporción", row=1, col=col)

    fig.update_layout(
        title="Análisis de Capacidad de Almacenamiento en Red de Hopfield (N = 25)",
        height=460,
        width=1100,
        legend={"orientation": "h", "y": -0.18, "x": 0.5, "xanchor": "center"},
    )
    save_fig(fig, out, "capacidad")
    return df_cap


def evaluate_noise_sweep(
    net: HopfieldNetwork,
    cfg: Dict[str, Any],
    out: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evalúa la recuperación para cada nivel de ruido y patrón; grafica métricas, ejemplos y energía."""
    rng = np.random.default_rng(cfg["seed"])
    noise_levels: List[float] = list(cfg["noise_levels"])
    trials: int = int(cfg["trials"])
    noise_type: str = cfg.get("type", "white")
    update_mode: str = cfg["update_mode"]
    max_iter: int = cfg["max_iterations"]

    trial_records = []
    energy_records = []
    # Guardamos un ejemplo representativo por nivel de ruido para la primera letra almacenada
    showcase_examples: List[Tuple[float, np.ndarray, np.ndarray, str, int, float]] = []

    for level in noise_levels:
        for pat_idx, name in enumerate(net.names):
            target = net.patterns[pat_idx]
            for t in range(trials):
                query = apply_noise(target, level, rng, noise_type=noise_type, shape=LETTER_SHAPE)
                res = net.predict(query, update_mode=update_mode, max_iterations=max_iter, rng=rng)
                outcome, matched_name, _ = net.classify_state(
                    res.final_state, res.converged, target_idx=pat_idx
                )
                ham_in = hamming_distance(query, target)
                ham_out = hamming_distance(res.final_state, target)
                pixel_acc = 1.0 - (ham_out / N_PIXELS)
                ov_target = float((target.astype(float) @ res.final_state.astype(float)) / N_PIXELS)

                trial_records.append({
                    "noise_level": level,
                    "flipped_pixels": ham_in,
                    "letter": name,
                    "trial": t,
                    "outcome": outcome,
                    "matched_pattern": matched_name or outcome,
                    "converged": res.converged,
                    "cycle_detected": res.cycle_detected,
                    "iterations": res.iterations,
                    "hamming_out": ham_out,
                    "pixel_accuracy": pixel_acc,
                    "target_overlap": ov_target,
                    "initial_energy": float(res.energy_history[0]),
                    "final_energy": float(res.energy_history[-1]),
                })

                # Guardamos la traza de energía del primer trial de cada letra y nivel de ruido
                if t == 0:
                    for step_idx, e_val in enumerate(res.energy_history):
                        energy_records.append({
                            "noise_level": level,
                            "letter": name,
                            "step": step_idx,
                            "energy": float(e_val),
                            "outcome": outcome,
                        })
                    if pat_idx == 0:
                        lbl = matched_name if matched_name else outcome
                        showcase_examples.append((
                            level,
                            query,
                            res.final_state,
                            lbl,
                            res.iterations,
                            float(res.energy_history[-1]),
                        ))

    df_trials = pd.DataFrame(trial_records)

    # Resumen por nivel de ruido
    summary_rows = []
    for level, grp in df_trials.groupby("noise_level", sort=True):
        n_tot = len(grp)
        counts = grp["outcome"].value_counts()
        summary_rows.append({
            "noise_level": level,
            "flipped_pixels": int(round(level * N_PIXELS)),
            "recovery_rate": round(counts.get("recovered", 0) / n_tot, 4),
            "other_stored_rate": round(counts.get("other_stored", 0) / n_tot, 4),
            "inverted_rate": round(counts.get("inverted", 0) / n_tot, 4),
            "spurious_rate": round(counts.get("spurious", 0) / n_tot, 4),
            "cycle_rate": round(counts.get("cycle", 0) / n_tot, 4),
            "mean_pixel_accuracy": round(float(grp["pixel_accuracy"].mean()), 4),
            "mean_iterations": round(float(grp["iterations"].mean()), 2),
            "std_iterations": round(float(grp["iterations"].std(ddof=0)), 2),
            "mean_final_energy": round(float(grp["final_energy"].mean()), 4),
        })
    df_summary = pd.DataFrame(summary_rows)
    save_csv(df_summary, out, "recuperacion_por_ruido")

    # Resumen desglosado por letra y nivel de ruido
    by_letter = (
        df_trials.groupby(["letter", "noise_level"], as_index=False)
        .agg(
            recovery_rate=("outcome", lambda s: round(float((s == "recovered").mean()), 4)),
            spurious_rate=("outcome", lambda s: round(float((s == "spurious").mean()), 4)),
            inverted_rate=("outcome", lambda s: round(float((s == "inverted").mean()), 4)),
            cycle_rate=("outcome", lambda s: round(float((s == "cycle").mean()), 4)),
            mean_pixel_accuracy=("pixel_accuracy", lambda s: round(float(s.mean()), 4)),
            mean_iterations=("iterations", lambda s: round(float(s.mean()), 2)),
        )
    )
    save_csv(by_letter, out, "recuperacion_por_letra")

    # Gráfico 1: Tasas de estado final e iteraciones vs nivel de ruido
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Distribución de estados alcanzados según nivel de ruido",
            "Iteraciones (barridos) promedio hasta convergencia",
        ),
    )
    outcome_meta = [
        ("recovery_rate", "Recuperado exacto (ξ^μ)", "#1e8449"),
        ("other_stored_rate", "Otra letra almacenada (ξ^ν)", "#2980b9"),
        ("spurious_rate", "Estado espurio (mezcla)", "#e67e22"),
        ("inverted_rate", "Patrón invertido (-ξ)", "#8e44ad"),
        ("cycle_rate", "Ciclo / no converge", "#c0392b"),
    ]
    for col_name, label, color in outcome_meta:
        fig.add_scatter(
            x=df_summary["noise_level"],
            y=df_summary[col_name],
            mode="lines+markers",
            name=label,
            line={"color": color, "width": 2.5},
            row=1,
            col=1,
        )

    fig.add_scatter(
        x=df_summary["noise_level"],
        y=df_summary["mean_iterations"],
        error_y={"type": "data", "array": df_summary["std_iterations"], "visible": True},
        mode="lines+markers",
        name="Iteraciones ± σ",
        line={"color": "#2c3e50", "width": 2.5},
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Nivel de ruido (fracción de píxeles invertidos)", row=1, col=1)
    fig.update_xaxes(title_text="Nivel de ruido (fracción de píxeles invertidos)", row=1, col=2)
    fig.update_yaxes(title_text="Proporción de consultas", range=[-0.05, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="Cantidad de barridos", row=1, col=2)
    fig.update_layout(
        title=f"Desempeño de Recuperación Asociativa ({update_mode}, letras={','.join(net.names)})",
        height=460,
        width=1120,
    )
    save_fig(fig, out, "recuperacion_por_ruido")

    # Gráfico 2: Energía en cada paso de la red
    df_energy = pd.DataFrame(energy_records)
    save_csv(df_energy, out, "energia_por_paso")

    fig_e = make_subplots(
        rows=1,
        cols=len(net.names),
        subplot_titles=[f"Consultas de letra {name}" for name in net.names],
        shared_yaxes=True,
    )
    palette = ["#1e8449", "#2980b9", "#16a085", "#f39c12", "#d35400", "#8e44ad", "#c0392b"]
    for pat_idx, name in enumerate(net.names):
        sub_e = df_energy[df_energy["letter"] == name]
        for lvl_idx, (lvl, grp) in enumerate(sub_e.groupby("noise_level", sort=True)):
            color = palette[lvl_idx % len(palette)]
            fig_e.add_scatter(
                x=grp["step"],
                y=grp["energy"],
                mode="lines+markers",
                name=f"ruido={lvl:g}",
                legendgroup=f"ruido={lvl:g}",
                showlegend=(pat_idx == 0),
                line={"color": color, "width": 2},
                row=1,
                col=pat_idx + 1,
            )
        e_pat = net.energy(net.patterns[pat_idx])
        fig_e.add_hline(
            y=e_pat,
            line_dash="dot",
            line_color="#333333",
            annotation_text=f"E({name})={e_pat:.2f}",
            annotation_position="bottom right",
            row=1,
            col=pat_idx + 1,
        )
        fig_e.update_xaxes(title_text="Paso (barrido)", dtick=1, row=1, col=pat_idx + 1)
    fig_e.update_yaxes(title_text="Energía E(s) = -½ sᵀWs", row=1, col=1)
    fig_e.update_layout(
        title=f"Evolución de la Energía por Paso ({update_mode})",
        height=440,
        width=max(290 * len(net.names), 900),
    )
    save_fig(fig_e, out, "energia_por_paso")

    # Gráfico 3: Grilla visual Original -> Consulta Ruidosa -> Recuperado por nivel de ruido
    n_cols = len(showcase_examples)
    if n_cols > 0:
        first_target = net.patterns[0]
        first_name = net.names[0]
        sub_titles = []
        for row_label in ("Consulta ruidosa", "Estado final red"):
            for lvl, q, fin, lbl, iters, e_fin in showcase_examples:
                if row_label == "Consulta ruidosa":
                    flips = hamming_distance(q, first_target)
                    sub_titles.append(f"Ruido {lvl:.0%} ({flips} px)<br>Entrada")
                else:
                    sub_titles.append(f"Final: {lbl} ({iters} it)<br>E = {e_fin:.2f}")

        fig_ex = make_subplots(
            rows=2,
            cols=n_cols,
            subplot_titles=sub_titles,
            vertical_spacing=0.14,
            horizontal_spacing=0.04,
        )
        for col_idx, (lvl, q, fin, lbl, iters, e_fin) in enumerate(showcase_examples, start=1):
            _add_pattern_heatmap(fig_ex, q, row=1, col=col_idx, diff_ref=first_target)
            _add_pattern_heatmap(fig_ex, fin, row=2, col=col_idx, diff_ref=first_target)

        fig_ex.update_layout(
            title=(
                f"Recuperación visual de la letra '{first_name}' según nivel de ruido "
                "(rojo = píxel distinto del patrón original)"
            ),
            height=540,
            width=max(210 * n_cols, 750),
        )
        save_fig(fig_ex, out, "ejemplos_recuperacion")

    return df_summary, by_letter


def analyze_spurious_states(net: HopfieldNetwork, cfg: Dict[str, Any], out: str) -> pd.DataFrame:
    """Ingresa patrones muy ruidosos e identifica/caracteriza estados espurios (mezclas e invertidos)."""
    rng = np.random.default_rng(cfg["seed"] + 997 if cfg["seed"] is not None else None)
    spurious_noise: float = float(cfg["spurious_noise"])
    noise_type: str = cfg.get("type", "white")
    update_mode: str = cfg["update_mode"]
    max_iter: int = cfg["max_iterations"]
    n_search = max(int(cfg["trials"]) * 4, 80)

    # Catálogo de atractores encontrados (clave = bytes del estado final)
    found: Dict[bytes, Dict[str, Any]] = {}

    # 1) Búsqueda mediante consultas muy ruidosas (nivel = spurious_noise)
    for idx, name in enumerate(net.names):
        pat = net.patterns[idx]
        for _ in range(n_search):
            q = apply_noise(pat, spurious_noise, rng, noise_type=noise_type, shape=LETTER_SHAPE)
            res = net.predict(q, update_mode=update_mode, max_iterations=max_iter, rng=rng)
            outcome, matched_name, _ = net.classify_state(res.final_state, res.converged, target_idx=idx)
            if outcome in ("spurious", "inverted", "cycle"):
                key = res.final_state.tobytes()
                if key not in found:
                    found[key] = {
                        "type": outcome,
                        "matched": matched_name,
                        "source_letter": name,
                        "noise_level": spurious_noise,
                        "query": q.copy(),
                        "final_state": res.final_state.copy(),
                        "energy": float(res.energy_history[-1]),
                        "iterations": res.iterations,
                        "converged": res.converged,
                        "is_fixed_point": net.is_fixed_point(res.final_state),
                        "overlaps": net.overlaps(res.final_state),
                        "count": 1,
                    }
                else:
                    found[key]["count"] += 1

    # 2) Si con el nivel configurado aún no apareció una mezcla espuria o un invertido,
    #    probamos también una consulta de combinación impar y una con ruido alto (0.76)
    #    para documentar ambos tipos de estados espurios clásicos de Hopfield.
    has_mixture = any(v["type"] == "spurious" for v in found.values())
    has_inverted = any(v["type"] == "inverted" for v in found.values())

    if not has_mixture and net.n_patterns >= 3:
        mix_raw = net.patterns[0].astype(int) + net.patterns[1].astype(int) + net.patterns[2].astype(int)
        q_mix = np.where(mix_raw >= 0, 1, -1).astype(np.int8)
        res = net.predict(q_mix, update_mode=update_mode, max_iterations=max_iter, rng=rng)
        outcome, matched_name, _ = net.classify_state(res.final_state, res.converged)
        if outcome in ("spurious", "cycle"):
            key = res.final_state.tobytes()
            if key not in found:
                found[key] = {
                    "type": outcome,
                    "matched": matched_name,
                    "source_letter": f"mix({','.join(net.names[:3])})",
                    "noise_level": round(hamming_distance(q_mix, net.patterns[0]) / N_PIXELS, 2),
                    "query": q_mix,
                    "final_state": res.final_state.copy(),
                    "energy": float(res.energy_history[-1]),
                    "iterations": res.iterations,
                    "converged": res.converged,
                    "is_fixed_point": net.is_fixed_point(res.final_state),
                    "overlaps": net.overlaps(res.final_state),
                    "count": 1,
                }

    if not has_inverted:
        q_inv = apply_noise(net.patterns[0], 0.80, rng, noise_type="white", shape=LETTER_SHAPE)
        res = net.predict(q_inv, update_mode=update_mode, max_iterations=max_iter, rng=rng)
        outcome, matched_name, _ = net.classify_state(res.final_state, res.converged)
        if outcome == "inverted":
            key = res.final_state.tobytes()
            if key not in found:
                found[key] = {
                    "type": outcome,
                    "matched": matched_name,
                    "source_letter": net.names[0],
                    "noise_level": 0.80,
                    "query": q_inv,
                    "final_state": res.final_state.copy(),
                    "energy": float(res.energy_history[-1]),
                    "iterations": res.iterations,
                    "converged": res.converged,
                    "is_fixed_point": net.is_fixed_point(res.final_state),
                    "overlaps": net.overlaps(res.final_state),
                    "count": 1,
                }

    # Ordenar priorizando mezclas espurias más frecuentes, luego invertidos, luego ciclos
    priority = {"spurious": 0, "inverted": 1, "cycle": 2}
    sorted_items = sorted(
        found.values(),
        key=lambda d: (priority.get(d["type"], 3), -d["count"], d["energy"]),
    )

    rows = []
    for idx, item in enumerate(sorted_items, start=1):
        ov_dict = {f"overlap_{name}": round(float(ov), 4) for name, ov in zip(net.names, item["overlaps"])}
        min_ham = min(hamming_distance(item["final_state"], p) for p in net.patterns)
        rows.append({
            "spurious_id": f"S{idx}",
            "state_type": item["type"],
            "matched_label": item["matched"] or "mezcla_espuria",
            "source_letter": item["source_letter"],
            "query_noise_level": item["noise_level"],
            "occurrences": item["count"],
            "energy": round(item["energy"], 4),
            "is_fixed_point": item["is_fixed_point"],
            "iterations": item["iterations"],
            "min_hamming_to_stored": min_ham,
            **ov_dict,
        })

    df_spurious = pd.DataFrame(rows)
    save_csv(df_spurious, out, "estados_espurios")

    # Graficar los primeros hasta 4 estados espurios encontrados (Fila 1: Consulta ruidosa, Fila 2: Estado espurio)
    showcase = sorted_items[:4]
    if showcase:
        n_cols = len(showcase)
        titles = []
        for row_idx in (1, 2):
            for idx, item in enumerate(showcase, start=1):
                if row_idx == 1:
                    titles.append(
                        f"Consulta #{idx} (origen: {item['source_letter']}, ruido={item['noise_level']:.0%})"
                    )
                else:
                    ov_str = " ".join(
                        f"{n}:{ov:+.2f}" for n, ov in zip(net.names, item["overlaps"])
                    )
                    tipo_lbl = (
                        f"Invertido ({item['matched']})"
                        if item["type"] == "inverted"
                        else ("Mezcla espuria" if item["type"] == "spurious" else "Ciclo")
                    )
                    titles.append(f"{tipo_lbl} | E={item['energy']:.2f}<br>m=[{ov_str}]")

        fig = make_subplots(
            rows=2,
            cols=n_cols,
            subplot_titles=titles,
            vertical_spacing=0.16,
            horizontal_spacing=0.05,
        )
        for col_idx, item in enumerate(showcase, start=1):
            _add_pattern_heatmap(fig, item["query"], row=1, col=col_idx)
            _add_pattern_heatmap(fig, item["final_state"], row=2, col=col_idx)

        fig.update_layout(
            title="Identificación de Estados Espurios a partir de Patrones Muy Ruidosos",
            height=560,
            width=max(260 * n_cols, 760),
        )
        save_fig(fig, out, "estados_espurios")

    return df_spurious


def run_letters_task(cfg: Dict[str, Any]) -> None:
    """Punto de entrada principal para el ejercicio 2.2 (reconocimiento de letras 5x5)."""
    sub = "letras" if cfg.get("task") == "all" else ""
    out = out_dir(cfg, sub)
    names, patterns = get_letter_patterns(cfg["letters"])
    net = HopfieldNetwork(patterns, names)

    # 1. Patrones almacenados y verificación de estabilidad
    df_stored = plot_stored_patterns(net, out)

    # 2. Análisis de ortogonalidad y capacidad de almacenamiento
    _, df_pairs = analyze_orthogonality(net, out)
    df_cap = analyze_capacity(cfg, out)

    # 3. Evaluación de recuperación frente a distintos niveles de ruido y trazas de energía
    df_noise, _ = evaluate_noise_sweep(net, cfg, out)

    # 4. Búsqueda y caracterización de estados espurios
    df_spurious = analyze_spurious_states(net, cfg, out)

    # Resumen por consola
    print(f"\n[letters] Letras almacenadas ({len(names)}): {', '.join(names)} (N = {N_PIXELS} neuronas)")
    for idx, name in enumerate(names):
        e = df_stored.loc[idx, "energy"]
        st = "estable" if df_stored.loc[idx, "is_fixed_point"] else "INESTABLE"
        print(f"\n  Letra {name} (E = {e:.2f}, {st}):")
        for line in format_ascii_grid(patterns[idx], LETTER_SHAPE).splitlines():
            print(f"    {line}")

    mean_ov = df_pairs["abs_overlap"].mean()
    max_row = df_pairs.iloc[0]
    print(
        f"\n[letters] Ortogonalidad: |m| medio = {mean_ov:.3f} | "
        f"par más correlacionado: {max_row['pair']} (m = {max_row['normalized_overlap']:+.2f}, "
        f"Hamming = {int(max_row['hamming_distance'])})"
    )
    print(
        f"[letters] Capacidad teórica (0.138·N): {THEORETICAL_CAPACITY:.2f} patrones "
        f"(con P={len(names)}: {int(df_stored['is_fixed_point'].sum())}/{len(names)} puntos fijos estables)"
    )

    print("\n[letters] Recuperación por nivel de ruido:")
    print(
        df_noise[
            [
                "noise_level",
                "flipped_pixels",
                "recovery_rate",
                "spurious_rate",
                "inverted_rate",
                "cycle_rate",
                "mean_iterations",
            ]
        ].to_string(index=False)
    )

    n_mix = int((df_spurious["state_type"] == "spurious").sum()) if not df_spurious.empty else 0
    n_inv = int((df_spurious["state_type"] == "inverted").sum()) if not df_spurious.empty else 0
    print(
        f"\n[letters] Estados espurios identificados (ruido={cfg['spurious_noise']:.0%}): "
        f"{len(df_spurious)} distintos ({n_mix} mezclas espurias, {n_inv} patrones invertidos)"
    )
    print(f"[letters] Gráficos HTML y tablas CSV guardados en: {out}/")
