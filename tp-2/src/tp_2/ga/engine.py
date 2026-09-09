import copy
import math
import random
from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np

from tp_2.ga.crossover import Crossover
from tp_2.ga.fitness import FitnessEvaluator
from tp_2.ga.individual import Individual
from tp_2.ga.mutation import Mutation
from tp_2.ga.selection import Selection
from tp_2.ga.stopping import StoppingCriteria
from tp_2.ga.survival import Survival
from tp_2.image.render import render_to_image
from tp_2.image.utils import ImageUtils


class GAEngine:

    def __init__(self, config: Any, evaluator: FitnessEvaluator):
        self.cfg       = config
        self.evaluator = evaluator
        self.stopping  = StoppingCriteria(config)
        self.history: List[Dict] = []   # métricas por generación

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
            t0    = self.cfg.get("boltzmann_t0", 100.0)
            decay = self.cfg.get("boltzmann_decay", self.cfg.get("boltzmann_k", 0.005))
            T     = t0 * math.exp(-decay * generation)
            return Selection.boltzmann(population, k, temperature=T)
        elif method in ("tournament_det", "det_tournament"):
            m = self.cfg.get("tournament_m", 3)
            return Selection.tournament_deterministic(population, k, m=m)
        elif method in ("tournament_prob", "prob_tournament"):
            p = self.cfg.get("tournament_threshold", 0.75)
            return Selection.tournament_probabilistic(population, k, threshold_p=p)
        elif method == "ranking":
            return Selection.ranking(population, k)
        elif method == "funsearch_priority":
            return Selection.funsearch_priority(population, k)
        elif method == "eoh_routing":
            return Selection.eoh_routing(population, k)
        else:
            print(f"[WARN] Método de selección desconocido: {method!r}. Usando roulette.")
            return Selection.roulette(population, k)

    def run(self) -> Individual:
        cfg = self.cfg

        pop_size       = cfg.get("pop_size",        50)
        num_triangles  = cfg.get("num_triangles",   30)
        children_size  = cfg.get("children_size",  pop_size)
        crossover_prob = cfg.get("crossover_prob",  0.8)
        mutation_prob  = cfg.get("mutation_prob",   0.1)
        add_triangle_prob = cfg.get("add_triangle_prob", 0.014136)
        p_tri          = cfg.get("p_tri",           0.193399)
        p_comp         = cfg.get("p_comp",          0.014537)
        max_gen        = cfg.get("max_generations", 1000)
        output_path    = cfg.get("output_path",     "output.png")
        save_interval  = cfg.get("save_interval",   100)
        save_frames    = cfg.get("save_frames",     True)
        frames_dir     = cfg.get("frames_dir",      "frames")
        parent_method  = cfg.get("parent_selection","roulette")
        surv_strategy  = cfg.get("survival_strategy","additive")
        surv_method    = cfg.get("survival_selection","elite")
        cross_method   = cfg.get("crossover",       "two_point")
        mut_method     = cfg.get("mutation",        "non_uniform")
        elitism        = cfg.get("elitism",          14)

        if save_frames:
            Path(frames_dir).mkdir(parents=True, exist_ok=True)

        population = [
            Individual.random_init(num_triangles) for _ in range(pop_size)
        ]
        self.evaluator.evaluate(population)

        best_candidate = max(population, key=lambda x: x.fitness if x.fitness is not None else -float("inf"))
        best = copy.deepcopy(best_candidate)
        generation = 0

        print(f"\n--- Evolución | pop={pop_size} | triángulos={num_triangles} ---")
        print(f"    cruza={cross_method} | mutación={mut_method}")
        print(f"    selección_padres={parent_method} | supervivencia={surv_strategy}/{surv_method}")

        rendered = render_to_image(best, self.evaluator.width, self.evaluator.height)
        ImageUtils.save_image(rendered, output_path)

        start_time = time.time()

        while True:
            generation += 1
            
            if random.random() < add_triangle_prob:
                num_triangles += 1
                for ind in population + [best]:
                    new_tri = np.array([[
                        random.random(), random.random(), random.random(),
                        random.random(), random.random(), random.random(),
                        random.uniform(0.0, 360.0), random.uniform(0.0, 130.0),
                        random.uniform(0.0, 100.0), random.uniform(0.05, 1.0)
                    ]], dtype=np.float32)
                    ind.genome = np.vstack([ind.genome, new_tri])
                    ind.fitness = None
                self.evaluator.evaluate(population + [best])

            best_fitness = best.fitness if best.fitness is not None else 0.0
            stop, reason = self.stopping.should_stop(generation, best_fitness)
            if stop:
                print(f"\n[FIN] Generación {generation}: {reason}")
                break

            parents = self._select(parent_method, population, children_size, generation)

            children: List[Individual] = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    p1, p2 = parents[i], parents[i + 1]
                    if random.random() < crossover_prob:
                        if cross_method == "one_point":
                            c1, c2 = Crossover.one_point(p1, p2)
                        elif cross_method == "uniform":
                            c1, c2 = Crossover.uniform(p1, p2)
                        elif cross_method == "annular":
                            c1, c2 = Crossover.annular(p1, p2)
                        elif cross_method == "adaptive_layer_spatial":
                            c1, c2 = Crossover.adaptive_layer_spatial(p1, p2)
                        else:
                            c1, c2 = Crossover.two_point(p1, p2)
                    else:
                        c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
                    children.extend([c1, c2])
                else:
                    children.append(copy.deepcopy(parents[i]))

            children = children[:children_size]

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
            self.evaluator.evaluate(children)

            def surv_selector(pool, k):
                return self._select(surv_method, pool, k, generation)

            population = Survival.select(
                population, children, pop_size,
                surv_strategy, surv_selector, elitism,
            )

            fits: List[float] = [ind.fitness for ind in population if ind.fitness is not None]
            
            gen_best = max(population, key=lambda x: x.fitness if x.fitness is not None else -float("inf"))
            
            gen_best_fit = gen_best.fitness if gen_best.fitness is not None else -float("inf")
            current_best_fit = best.fitness if best.fitness is not None else -float("inf")
            
            improved = gen_best_fit > current_best_fit
            if improved:
                best = copy.deepcopy(gen_best)

            mean_fit = float(np.mean(fits))
            worst_fit = float(np.min(fits))
            std_fit = float(np.std(fits))
            elapsed_sec = time.time() - start_time

            # Despeje exacto de Delta E desde la fórmula normalizada [0.0, 1.0]: fitness = 1 - (delta_e / 100)
            best_delta_e = (1.0 - gen_best_fit) * 100.0 if gen_best_fit >= 0 else float("inf")
            mean_delta_e = (1.0 - mean_fit) * 100.0 if mean_fit >= 0 else float("inf")

            self.history.append({
                "generation":       generation,
                "elapsed_time_sec": elapsed_sec,
                "best":             best.fitness if best.fitness is not None else 0.0,
                "mean":             mean_fit,
                "std":              std_fit,
                "worst":            worst_fit,
                "best_error":       best_delta_e,
                "mean_error":       mean_delta_e,
            })

            if generation % save_interval == 0 or improved or generation == 1:
                tag = " [MEJORA]" if improved else ""
                print(
                    f"Gen {generation:5d} | "
                    f"mejor={best.fitness:.4f} | "
                    f"media={self.history[-1]['mean']:.4f} | "
                    f"std={self.history[-1]['std']:.4f}"
                    f"{tag}"
                )
                rendered = render_to_image(
                    best, self.evaluator.width, self.evaluator.height
                )
                ImageUtils.save_image(rendered, output_path)
                if save_frames:
                    ImageUtils.save_image(
                        rendered,
                        str(Path(frames_dir) / f"gen_{generation:05d}.png")
                    )

        final_rendered = render_to_image(best, self.evaluator.width, self.evaluator.height)
        ImageUtils.save_image(final_rendered, output_path)

        return best
