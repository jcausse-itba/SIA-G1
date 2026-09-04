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