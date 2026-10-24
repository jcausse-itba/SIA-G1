"""Modelo de Red de Hopfield (aprendizaje hebbiano, dinámica sincrónica/asincrónica y energía)."""

__all__ = [
    "HopfieldResult",
    "HopfieldNetwork",
    "apply_noise",
    "hamming_distance",
]

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class HopfieldResult:
    """Resultado de una consulta asociativa en la red de Hopfield."""

    initial_state: np.ndarray        # Estado inicial (consulta ruidosa), shape (N,)
    final_state: np.ndarray          # Estado final alcanzado, shape (N,)
    states_history: List[np.ndarray] # Estados s(0), s(1), ..., s(T) por barrido
    energy_history: np.ndarray       # Energía E(s(0)), E(s(1)), ..., E(s(T))
    iterations: int                  # Cantidad de barridos ejecutados
    converged: bool                  # True si alcanzó un punto fijo s(t) == s(t-1)
    cycle_detected: bool             # True si entró en un ciclo límite de período 2


class HopfieldNetwork:
    """Red de Hopfield con regla de Hebb sobre patrones bipolares {-1, +1}.

    Matriz de pesos:
        W = (1 / N) * sum_{mu=1}^{P} xi^{mu} (xi^{mu})^T ,  con W_{ii} = 0.
    Función de energía de Lyapunov:
        E(s) = -1/2 * s^T W s.
    """

    def __init__(self, patterns: np.ndarray, names: Sequence[str] | None = None) -> None:
        patterns = np.asarray(patterns, dtype=np.int8)
        if patterns.ndim != 2:
            raise ValueError(f"Expected 2D array of shape (P, N), got {patterns.shape}")
        if not np.isin(patterns, (-1, 1)).all():
            raise ValueError("All pattern entries must be in {-1, +1}.")

        self.patterns: np.ndarray = patterns.copy()
        self.n_patterns, self.n_neurons = self.patterns.shape
        self.names: List[str] = (
            list(names) if names is not None else [f"P{i}" for i in range(self.n_patterns)]
        )

        # Regla de Hebb: W = (1/N) * K * K^T (donde K tiene a los patrones por columna)
        Xf = self.patterns.astype(np.float64)
        self.W: np.ndarray = (Xf.T @ Xf) / self.n_neurons
        np.fill_diagonal(self.W, 0.0)

    def energy(self, state: np.ndarray) -> float:
        """Calcula E(s) = -0.5 * s^T W s."""
        s = np.asarray(state, dtype=np.float64)
        return float(-0.5 * (s @ (self.W @ s)))

    def local_field(self, state: np.ndarray) -> np.ndarray:
        """Campo local h = W s."""
        return self.W @ np.asarray(state, dtype=np.float64)

    def is_fixed_point(self, state: np.ndarray) -> bool:
        """Verifica si state es un punto fijo exacto bajo actualización determinística."""
        s = np.asarray(state, dtype=np.int8)
        h = self.local_field(s)
        s_next = np.where(h > 1e-12, 1, np.where(h < -1e-12, -1, s))
        return bool(np.array_equal(s_next, s))

    def overlaps(self, state: np.ndarray) -> np.ndarray:
        """Solapamiento normalizado m_mu(s) = (1/N) * xi^mu · s en [-1, 1] para cada patrón."""
        s = np.asarray(state, dtype=np.float64)
        return (self.patterns.astype(np.float64) @ s) / self.n_neurons

    def orthogonality_matrix(self) -> np.ndarray:
        """Matriz de solapamientos (productos internos normalizados) entre patrones almacenados."""
        Xf = self.patterns.astype(np.float64)
        return (Xf @ Xf.T) / self.n_neurons

    def dot_product_matrix(self) -> np.ndarray:
        """Matriz de productos internos <xi^mu, xi^nu> en [-N, N]."""
        Xi = self.patterns.astype(np.int32)
        return Xi @ Xi.T

    def predict(
        self,
        query: np.ndarray,
        update_mode: str = "asynchronous",
        max_iterations: int = 100,
        rng: np.random.Generator | None = None,
    ) -> HopfieldResult:
        """Evoluciona la red desde `query` hasta converger a un atractor o agotar iteraciones."""
        if rng is None:
            rng = np.random.default_rng()

        s = np.asarray(query, dtype=np.int8).copy()
        states_hist: List[np.ndarray] = [s.copy()]
        energies: List[float] = [self.energy(s)]

        converged = False
        cycle_detected = False
        cycle_steps_remaining: int | None = None
        it = 0

        for it in range(1, max_iterations + 1):
            prev_s = s.copy()

            if update_mode == "synchronous":
                h = self.W @ s.astype(np.float64)
                s = np.where(h > 1e-12, 1, np.where(h < -1e-12, -1, s)).astype(np.int8)
            elif update_mode == "asynchronous":
                order = rng.permutation(self.n_neurons)
                for idx in order:
                    h_i = float(self.W[idx] @ s)
                    if h_i > 1e-12:
                        s[idx] = 1
                    elif h_i < -1e-12:
                        s[idx] = -1
            else:
                raise ValueError(f"Unknown update_mode: {update_mode!r}")

            states_hist.append(s.copy())
            energies.append(self.energy(s))

            # Punto fijo alcanzado
            if np.array_equal(s, prev_s) and self.is_fixed_point(s):
                converged = True
                break

            # Detección de ciclo de período 2 en modo sincrónico: s(t) == s(t-2) con s(t) != s(t-1)
            if (
                update_mode == "synchronous"
                and len(states_hist) >= 3
                and np.array_equal(s, states_hist[-3])
            ):
                cycle_detected = True
                if cycle_steps_remaining is None:
                    # Dejar 3 pasos adicionales para visualizar la oscilación en el gráfico de energía
                    cycle_steps_remaining = 3
                else:
                    cycle_steps_remaining -= 1
                    if cycle_steps_remaining <= 0:
                        break

        return HopfieldResult(
            initial_state=states_hist[0],
            final_state=s,
            states_history=states_hist,
            energy_history=np.array(energies, dtype=np.float64),
            iterations=it,
            converged=converged,
            cycle_detected=cycle_detected,
        )

    def classify_state(
        self,
        state: np.ndarray,
        converged: bool,
        target_idx: int | None = None,
    ) -> Tuple[str, str | None, int | None]:
        """Clasifica el estado final de la red.

        Retorna (outcome, matched_name, matched_idx):
        - "recovered": coincide exactamente con el patrón objetivo (o con algún patrón si target_idx es None)
        - "other_stored": coincide exactamente con otro patrón almacenado distinto del objetivo
        - "inverted": coincide exactamente con el opuesto (-xi^mu) de un patrón almacenado
        - "spurious": punto fijo que no es ninguno de los +-xi^mu (estado espurio de mezcla/mínimo local)
        - "cycle": no convergió a un punto fijo (ciclo límite o tope de iteraciones)
        """
        for idx, pat in enumerate(self.patterns):
            if np.array_equal(state, pat):
                if target_idx is None or idx == target_idx:
                    return "recovered", self.names[idx], idx
                return "other_stored", self.names[idx], idx

        for idx, pat in enumerate(self.patterns):
            if np.array_equal(state, -pat):
                return "inverted", f"-{self.names[idx]}", idx

        if not converged:
            return "cycle", None, None

        return "spurious", None, None


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Cantidad de posiciones en las que difieren dos vectores bipolares."""
    return int(np.count_nonzero(a != b))


def apply_noise(
    pattern: np.ndarray,
    level: float,
    rng: np.random.Generator,
    noise_type: str = "white",
    shape: Tuple[int, int] | None = None,
) -> np.ndarray:
    """Aplica ruido invirtiendo exactamente round(level * N) píxeles del patrón.

    Tipos de ruido soportados:
    - "white": selección uniforme sin reemplazo de round(level * N) píxeles.
    - "finger": mancha conexa/circular centrada en un punto aleatorio (simula dedo/mancha).
    - "perlin": campo aleatorio espacialmente correlacionado (filtro gaussiano multi-escala).
    """
    noisy = np.asarray(pattern, dtype=np.int8).copy()
    n = noisy.size
    n_flip = int(round(level * n))
    if n_flip <= 0:
        return noisy
    if n_flip >= n:
        return -noisy

    if noise_type == "white" or shape is None:
        idx = rng.choice(n, size=n_flip, replace=False)
        noisy[idx] *= -1
        return noisy

    rows, cols = shape
    if noise_type == "finger":
        r0 = rng.uniform(0, rows - 1)
        c0 = rng.uniform(0, cols - 1)
        rr, cc = np.indices((rows, cols))
        dist = np.sqrt((rr - r0) ** 2 + (cc - c0) ** 2) + rng.normal(0, 0.35, size=(rows, cols))
        idx = np.argsort(dist.ravel())[:n_flip]
        noisy[idx] *= -1
        return noisy

    if noise_type == "perlin":
        field = np.zeros((rows, cols), dtype=np.float64)
        for sigma in (1.0, 2.0, 4.0):
            raw = rng.standard_normal((rows, cols))
            field += gaussian_filter(raw, sigma=sigma)
        idx = np.argsort(field.ravel())[:n_flip]
        noisy[idx] *= -1
        return noisy

    raise ValueError(f"Unknown noise_type: {noise_type!r}")
