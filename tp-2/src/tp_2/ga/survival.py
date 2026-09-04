from typing import List, Callable
from tp_2.ga.individual import Individual

class Survival:
    """Estrategias de supervivencia para conformar la nueva generación."""

    @staticmethod
    def select(
        parents: List[Individual],
        children: List[Individual],
        population_size: int,
        strategy: str,
        selector_func: Callable[[List[Individual], int], List[Individual]]
    ) -> List[Individual]:
        
        if strategy == "additive":
            # Mu + Lambda: Selección sobre la unión de padres e hijos
            pool = parents + children
            return selector_func(pool, population_size)
            
        elif strategy == "exclusive":
            # Mu, Lambda: Selección únicamente sobre la descendencia
            if len(children) >= population_size:
                return selector_func(children, population_size)
            else:
                # Si faltan hijos, rellenar con los mejores padres
                needed = population_size - len(children)
                sorted_parents = sorted(parents, key=lambda x: x.fitness, reverse=True)
                return children + sorted_parents[:needed]
        else:
            raise ValueError(f"Estrategia de supervivencia desconocida: {strategy}")