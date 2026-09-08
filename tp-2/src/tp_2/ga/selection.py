import math
import random
from typing import List
import numpy as np
from tp_2.ga.individual import Individual

class Selection:
    """Métodos de selección para la población de Algoritmos Genéticos."""

    @staticmethod
    def elite(population: List[Individual], k: int) -> List[Individual]:
        """Selección por Elitismo: retorna los k individuos con mayor fitness."""
        sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
        return sorted_pop[:k]

    @staticmethod
    def roulette(population: List[Individual], k: int) -> List[Individual]:
        """Selección por Ruleta: probabilidad proporcional al fitness."""
        total_fitness = sum(ind.fitness for ind in population)
        if total_fitness == 0:
            return random.choices(population, k=k)

        probs = [ind.fitness / total_fitness for ind in population]
        cum_probs = np.cumsum(probs)

        selected = []
        for _ in range(k):
            r = random.random()
            idx = np.searchsorted(cum_probs, r)
            selected.append(population[min(idx, len(population) - 1)])
        return selected

    @staticmethod
    def universal(population: List[Individual], k: int) -> List[Individual]:
        """Stochastic Universal Sampling (SUS): reduce varianza respecto a la ruleta."""
        total_fitness = sum(ind.fitness for ind in population)
        if total_fitness == 0:
            return random.choices(population, k=k)

        probs = [ind.fitness / total_fitness for ind in population]
        cum_probs = np.cumsum(probs)

        step = 1.0 / k
        start = random.uniform(0, step)
        pointers = [start + i * step for i in range(k)]

        selected = []
        for p in pointers:
            idx = np.searchsorted(cum_probs, p)
            selected.append(population[min(idx, len(population) - 1)])
        return selected

    @staticmethod
    def boltzmann(population: List[Individual], k: int, temperature: float) -> List[Individual]:
        """Selección de Boltzmann: escala el fitness según la temperatura actual."""
        # Evitar desbordamiento numérico restando el máximo
        max_fit = max(ind.fitness for ind in population)
        exp_values = [math.exp((ind.fitness - max_fit) / temperature) for ind in population]
        sum_exp = sum(exp_values)

        probs = [val / sum_exp for val in exp_values]
        cum_probs = np.cumsum(probs)

        selected = []
        for _ in range(k):
            r = random.random()
            idx = np.searchsorted(cum_probs, r)
            selected.append(population[min(idx, len(population) - 1)])
        return selected

    @staticmethod
    def tournament_deterministic(population: List[Individual], k: int, m: int = 3) -> List[Individual]:
        """Torneo Determinístico: elige el mejor individuo entre m seleccionados al azar."""
        selected = []
        for _ in range(k):
            competitors = random.sample(population, m)
            best = max(competitors, key=lambda ind: ind.fitness)
            selected.append(best)
        return selected

    @staticmethod
    def tournament_probabilistic(population: List[Individual], k: int, threshold_p: float = 0.75) -> List[Individual]:
        """Torneo Probabilístico (m=2): gana el mejor con probabilidad p, o el peor con (1-p)."""
        selected = []
        for _ in range(k):
            ind1, ind2 = random.sample(population, 2)
            best, worst = (ind1, ind2) if ind1.fitness >= ind2.fitness else (ind2, ind1)
            
            if random.random() < threshold_p:
                selected.append(best)
            else:
                selected.append(worst)
        return selected

    @staticmethod
    def ranking(population: List[Individual], k: int) -> List[Individual]:
        """Selección por Ranking: asigna probabilidades según la posición ordenada."""
        sorted_pop = sorted(population, key=lambda ind: ind.fitness)
        n = len(sorted_pop)
        
        # Asignación lineal de probabilidades basada en rango (1 a N)
        total_rank = n * (n + 1) / 2
        probs = [(i + 1) / total_rank for i in range(n)]
        cum_probs = np.cumsum(probs)

        selected = []
        for _ in range(k):
            r = random.random()
            idx = np.searchsorted(cum_probs, r)
            selected.append(sorted_pop[min(idx, n - 1)])
        return selected

    @staticmethod
    def funsearch_priority(population: List[Individual], k: int, temperature: float = 3.0333557176709616, length_penalty_weight: float = 0.0) -> List[Individual]:
        """Selección por Prioridad de FunSearch: combina fitness escalado por temperatura con penalización opcional por longitud."""

        max_fit = max(ind.fitness for ind in population)
        priorities = []
        for ind in population:
            fit_score = math.exp((ind.fitness - max_fit) / temperature)
            length_penalty = 1.0
            if length_penalty_weight > 0:
                if hasattr(ind, 'genome'):
                    length = len(ind.genome)
                else:
                    length = getattr(ind, 'length', len(ind) if hasattr(ind, '__len__') else 0)
                length_penalty = 1.0 / (1.0 + length_penalty_weight * length)
            priorities.append(fit_score * length_penalty)

        total_priority = sum(priorities)
        if total_priority == 0:
            return random.choices(population, k=k)

        probs = [p / total_priority for p in priorities]
        cum_probs = np.cumsum(probs)

        selected = []
        for _ in range(k):
            r = random.random()
            idx = np.searchsorted(cum_probs, r)
            selected.append(population[min(idx, len(population) - 1)])
        return selected

    @staticmethod
    def eoh_routing(population: List[Individual], k: int, elite_ratio: float = 0.2, routing_prob: float = 0.8) -> List[Individual]:
        """Selección por Enrutamiento de EoH (Evolution of Heuristics): enruta la selección equilibrando la elite y la exploración general."""
        sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
        n = len(sorted_pop)
        num_elite = max(1, int(n * elite_ratio))

        elites = sorted_pop[:num_elite]
        others = sorted_pop[num_elite:] if num_elite < n else sorted_pop

        selected = []
        for _ in range(k):
            if random.random() < routing_prob:
                selected.append(random.choice(elites))
            else:
                selected.append(random.choice(others))
        return selected
