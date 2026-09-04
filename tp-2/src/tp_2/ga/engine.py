import copy
import math
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

from tp_2.ga.crossover import Crossover
from tp_2.ga.fitness import FitnessEvaluator
from tp_2.ga.individual import Individual
from tp_2.ga.mutation import Mutation
from tp_2.ga.selection import Selection
from tp_2.ga.stopping import StoppingCriteria
from tp_2.ga.survival import Survival
from tp_2.image.render import render_individual
from tp_2.image.utils import ImageUtils
from tp_2.metrics.plotting import plot_fitness_curve, plot_diversity, save_metrics_csv


class GAEngine:

    def __init__(self, config: Any, evaluator: FitnessEvaluator):
        self.cfg       = config
        self.evaluator = evaluator
        self.stopping  = StoppingCriteria(config)
        self.history: List[Dict] = []   # métricas por generación

    # ------------------------------------------------------------------
    # Dispatcher de selección (incluye temperatura para Boltzmann)
    # ------------------------------------------------------------------
    def _select(
        self,
        method: str,
        population: List[Individual],
        k: int,
        generation: int = 1,
    ) -> List[Individual]:
        if method == "elite":
            return Selection.elite(population, k)
        elif method == "roulette":
            return Selection.roulette(population, k)
        elif method == "universal":
            return Selection.universal(population, k)
        elif method == "boltzmann":
            t0    = getattr(self.cfg, "boltzmann_t0",    100.0)
            decay = getattr(self.cfg, "boltzmann_decay",   0.005)
            T     = t0 * math.exp(-decay * generation)
            return Selection.boltzmann(population, k, temperature=T)
        elif method in ("tournament_det", "det_tournament"):
            m = getattr(self.cfg, "tournament_m", 3)
            return Selection.tournament_deterministic(population, k, m=m)
        elif method in ("tournament_prob", "prob_tournament"):
            p = getattr(self.cfg, "tournament_threshold", 0.75)
            return Selection.tournament_probabilistic(population, k, threshold_p=p)
        elif method == "ranking":
            return Selection.ranking(population, k)
        else:
            print(f"[WARN] Método de selección desconocido: {method!r}. Usando roulette.")
            return Selection.roulette(population, k)

    # ------------------------------------------------------------------
    # Loop principal
    # ------------------------------------------------------------------
    def run(self) -> Individual:
        cfg = self.cfg

        pop_size       = getattr(cfg, "pop_size",        50)
        num_triangles  = getattr(cfg, "num_triangles",   30)
        children_size  = getattr(cfg, "children_size",  pop_size)
        crossover_prob = getattr(cfg, "crossover_prob",  0.8)
        mutation_prob  = getattr(cfg, "mutation_prob",   0.1)   # p_ind
        p_tri          = getattr(cfg, "p_tri",           0.3)
        p_comp         = getattr(cfg, "p_comp",          0.2)
        max_gen        = getattr(cfg, "max_generations", 1000)
        output_path    = getattr(cfg, "output_path",     "output.png")
        save_interval  = getattr(cfg, "save_interval",   10)
        save_frames    = getattr(cfg, "save_frames",     True)
        frames_dir     = getattr(cfg, "frames_dir",      "frames")
        parent_method  = getattr(cfg, "parent_selection","roulette")
        surv_strategy  = getattr(cfg, "survival_strategy","additive")
        surv_method    = getattr(cfg, "survival_selection","elite")
        cross_method   = getattr(cfg, "crossover",       "two_point")
        mut_method     = getattr(cfg, "mutation",        "non_uniform")
        elitism        = getattr(cfg, "elitism",          1)

        if save_frames:
            Path(frames_dir).mkdir(parents=True, exist_ok=True)

        # 1. Población inicial
        population = [
            Individual.random_init(num_triangles) for _ in range(pop_size)
        ]
        for ind in population:
            self.evaluator.evaluate(ind)

        best = max(population, key=lambda x: x.fitness)
        generation = 0

        print(f"\n--- Evolución | pop={pop_size} | triángulos={num_triangles} ---")
        print(f"    cruza={cross_method} | mutación={mut_method}")
        print(f"    selección_padres={parent_method} | supervivencia={surv_strategy}/{surv_method}")

        # Render inicial
        rendered = render_individual(best, self.evaluator.width, self.evaluator.height)
        ImageUtils.save_image(rendered, output_path)

        while True:
            generation += 1

            stop, reason = self.stopping.should_stop(generation, best.fitness)
            if stop:
                print(f"\n[FIN] Generación {generation}: {reason}")
                break

            # 2. Selección de padres
            parents = self._select(parent_method, population, children_size, generation)

            # 3. Cruza
            children: List[Individual] = []
            for i in range(0, len(parents) - 1, 2):
                p1, p2 = parents[i], parents[i + 1]
                if random.random() < crossover_prob:
                    if cross_method == "one_point":
                        c1, c2 = Crossover.one_point(p1, p2)
                    elif cross_method == "uniform":
                        c1, c2 = Crossover.uniform(p1, p2)
                    elif cross_method == "annular":
                        c1, c2 = Crossover.annular(p1, p2)
                    else:  # two_point (default)
                        c1, c2 = Crossover.two_point(p1, p2)
                else:
                    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
                children.extend([c1, c2])

            # 4. Mutación jerárquica
            for i, child in enumerate(children):
                children[i] = Mutation.apply(
                    child,
                    p_ind=mutation_prob,
                    method=mut_method,
                    generation=generation,
                    max_generations=max_gen,
                    p_tri=p_tri,
                    p_comp=p_comp,
                )
                self.evaluator.evaluate(children[i])

            # 5. Supervivencia
            def surv_selector(pool, k):
                return self._select(surv_method, pool, k, generation)

            population = Survival.select(
                population, children, pop_size,
                surv_strategy, surv_selector, elitism,
            )

            # 6. Métricas
            fits = [ind.fitness for ind in population]
            gen_best = max(population, key=lambda x: x.fitness)
            improved = gen_best.fitness > best.fitness
            if improved:
                best = gen_best

            self.history.append({
                "generation":  generation,
                "best":        best.fitness,
                "mean":        float(np.mean(fits)),
                "std":         float(np.std(fits)),
                "worst":       float(np.min(fits)),
            })

            # 7. Logging y guardado
            if generation % save_interval == 0 or improved or generation == 1:
                tag = " [MEJORA]" if improved else ""
                print(
                    f"Gen {generation:5d} | "
                    f"mejor={best.fitness:.4f} | "
                    f"media={self.history[-1]['mean']:.4f} | "
                    f"std={self.history[-1]['std']:.4f}"
                    f"{tag}"
                )
                rendered = render_individual(
                    best, self.evaluator.width, self.evaluator.height
                )
                ImageUtils.save_image(rendered, output_path)
                if save_frames:
                    ImageUtils.save_image(
                        rendered,
                        str(Path(frames_dir) / f"gen_{generation:05d}.png")
                    )

        # 8. Métricas finales
        metrics_dir = getattr(cfg, "metrics_dir", "metrics")
        label = (
            f"{cross_method}_{mut_method}_{parent_method}_{surv_strategy}"
        )
        plot_fitness_curve(
            self.history,
            path=f"{metrics_dir}/fitness_curve.png",
            title=f"Fitness — {label}",
        )
        plot_diversity(
            self.history,
            path=f"{metrics_dir}/diversity.png",
            title=f"Diversidad — {label}",
        )
        save_metrics_csv(self.history, path=f"{metrics_dir}/metrics.csv")

        return best