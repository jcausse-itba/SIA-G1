import copy
from typing import Callable, List
from tp_2.ga.individual import Individual


class Survival:

  @staticmethod
  def select(
      parents: List[Individual],
      children: List[Individual],
      pop_size: int,
      strategy: str,
      selector: Callable[[List[Individual], int], List[Individual]],
      elitism: int = 0,
  ) -> List[Individual]:
    # Pool de selección según la estrategia (aditiva: mu + lambda, exclusiva: lambda)
    if strategy == "additive":
      pool = parents + children
    else:  # subtractive / exclusive
      pool = children if len(children) >= pop_size else parents + children

    # Separar la élite si está configurada
    elite: List[Individual] = []
    if elitism > 0:
      # La élite siempre se extrae de la combinación de padres e hijos
      full_pool = sorted(
          parents + children, key=lambda ind: ind.fitness, reverse=True
      )
      elite = [copy.deepcopy(ind) for ind in full_pool[:elitism]]

    remaining_k = pop_size - len(elite)
    if remaining_k <= 0:
      return elite[:pop_size]

    # Seleccionar el resto mediante el método configurado
    selected = selector(pool, remaining_k)
    return elite + selected