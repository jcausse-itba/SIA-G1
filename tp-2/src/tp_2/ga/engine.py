import copy
import random
import time
from pathlib import Path
from typing import Any, List

from tp_2.ga.crossover import Crossover
from tp_2.ga.fitness import FitnessEvaluator
from tp_2.ga.individual import Individual
from tp_2.ga.mutation import Mutation
from tp_2.ga.selection import Selection
from tp_2.ga.stopping import StoppingCriteria
from tp_2.ga.survival import Survival
from tp_2.image.render import render_individual
from tp_2.image.utils import ImageUtils


class GAEngine:

  def __init__(self, config: Any, evaluator: FitnessEvaluator):
    self.cfg = config
    self.evaluator = evaluator
    self.stopping = StoppingCriteria(config)

  def _select(
      self, method_name: str, population: List[Individual], k: int
  ) -> List[Individual]:
    if method_name == "elite":
      return Selection.elite(population, k)
    elif method_name == "roulette":
      return Selection.roulette(population, k)
    elif method_name == "universal":
      return Selection.universal(population, k)
    elif method_name == "boltzmann":
      t0 = getattr(self.cfg, "boltzmann_t0", 100.0)
      return Selection.boltzmann(population, k, temperature=t0)
    elif method_name in ("det_tournament", "tournament_det"):
      m = getattr(self.cfg, "tournament_m", 3)
      return Selection.tournament_deterministic(population, k, m=m)
    elif method_name in ("prob_tournament", "tournament_prob"):
      p = getattr(self.cfg, "tournament_threshold", 0.75)
      return Selection.tournament_probabilistic(population, k, threshold_p=p)
    elif method_name == "ranking":
      return Selection.ranking(population, k)
    else:
      return Selection.roulette(population, k)

  def run(self) -> Individual:
    pop_size = getattr(self.cfg, "pop_size", 50)
    num_triangles = getattr(self.cfg, "num_triangles", 30)
    children_size = getattr(self.cfg, "children_size", pop_size)
    crossover_prob = getattr(self.cfg, "crossover_prob", 0.8)
    mutation_prob = getattr(self.cfg, "mutation_prob", 0.1)
    output_path = getattr(self.cfg, "output_path", "output.png")

    # Crear directorio para cuadros intermedios si quisiéramos guardar secuencia
    frames_dir = Path("frames")
    frames_dir.mkdir(exist_ok=True)

    # 1. Inicializar población
    population = [
        Individual.random_init(num_triangles) for _ in range(pop_size)
    ]
    for ind in population:
      self.evaluator.evaluate(ind)

    generation = 0
    best_ind = max(population, key=lambda x: x.fitness)

    print(
        f"\n--- Inicio de Evolución (Población: {pop_size}, Triángulos:"
        f" {num_triangles}) ---"
    )

    # Guardar estado inicial (generación 0)
    initial_render = render_individual(
        best_ind, self.evaluator.width, self.evaluator.height
    )
    ImageUtils.save_image(initial_render, output_path)

    while True:
      generation += 1

      # Verificar condición de corte
      stop, reason = self.stopping.should_stop(generation, best_ind.fitness)
      if stop:
        print(f"\n[Fin de la Evolución] Generación {generation}: {reason}")
        break

      # 2. Selección de Padres
      parent_method = getattr(self.cfg, "parent_selection", "roulette")
      parents = self._select(parent_method, population, children_size)

      # 3. Cruza
      cross_method = getattr(self.cfg, "crossover", "two_point")
      children = []
      for i in range(0, len(parents) - 1, 2):
        p1, p2 = parents[i], parents[i + 1]
        if random.random() < crossover_prob:
          if cross_method == "one_point":
            c1, c2 = Crossover.one_point(p1, p2)
          elif cross_method == "uniform":
            c1, c2 = Crossover.uniform(p1, p2)
          elif cross_method == "annular":
            c1, c2 = Crossover.annular(p1, p2)
          else:
            c1, c2 = Crossover.two_point(p1, p2)
        else:
          c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
        children.extend([c1, c2])

      # 4. Mutación
      mut_method = getattr(self.cfg, "mutation", "multigene_limited")
      max_gen = getattr(self.cfg, "max_generations", 1000)
      for i in range(len(children)):
        children[i] = Mutation.apply(
            children[i], mutation_prob, mut_method, generation, max_gen
        )
        self.evaluator.evaluate(children[i])

      # 5. Supervivencia
      surv_strategy = getattr(self.cfg, "survival_strategy", "additive")
      surv_method = getattr(self.cfg, "survival_selection", "elite")

      def surv_selector(pool, k):
        return self._select(surv_method, pool, k)

      population = Survival.select(
          population, children, pop_size, surv_strategy, surv_selector
      )

      # Actualizar mejor individuo
      current_best = max(population, key=lambda x: x.fitness)
      improved = current_best.fitness > best_ind.fitness

      if improved:
        best_ind = current_best

      # Guardar imagen actualizada cada 5 generaciones o cuando haya mejora
      if generation % 50 == 0 or generation == 1:
        print(
            f"Gen {generation:4d} | Mejor Fitness: {best_ind.fitness:.4f}"
            f" {'[¡Mejora!]' if improved else ''}"
        )
        rendered = render_individual(
            best_ind, self.evaluator.width, self.evaluator.height
        )

        # Sobrescribir output principal
        ImageUtils.save_image(rendered, output_path)

        # Guardar frame histórico opcional
        ImageUtils.save_image(rendered, str(frames_dir / f"gen_{generation:04d}.png"))

    return best_ind