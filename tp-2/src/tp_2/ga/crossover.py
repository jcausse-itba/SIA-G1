import random
from typing import List, Tuple
from tp_2.ga.individual import Individual, Triangle

class Crossover:
    """Métodos de cruza para el motor genético (a nivel de lista de triángulos)."""

    @staticmethod
    def one_point(p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        n = len(p1.triangles)
        point = random.randint(1, n - 1)
        
        c1_triangles = p1.triangles[:point] + p2.triangles[point:]
        c2_triangles = p2.triangles[:point] + p1.triangles[point:]
        
        return Individual(c1_triangles), Individual(c2_triangles)

    @staticmethod
    def two_point(p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        n = len(p1.triangles)
        pt1, pt2 = sorted(random.sample(range(1, n), 2))
        
        c1 = p1.triangles[:pt1] + p2.triangles[pt1:pt2] + p1.triangles[pt2:]
        c2 = p2.triangles[:pt1] + p1.triangles[pt1:pt2] + p2.triangles[pt2:]
        
        return Individual(c1), Individual(c2)

    @staticmethod
    def uniform(p1: Individual, p2: Individual, p: float = 0.5) -> Tuple[Individual, Individual]:
        c1_tri, c2_tri = [], []
        for t1, t2 in zip(p1.triangles, p2.triangles):
            if random.random() < p:
                c1_tri.append(t1)
                c2_tri.append(t2)
            else:
                c1_tri.append(t2)
                c2_tri.append(t1)
        return Individual(c1_tri), Individual(c2_tri)

    @staticmethod
    def annular(p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        n = len(p1.triangles)
        start = random.randint(0, n - 1)
        length = random.randint(1, n // 2)
        
        indices = [(start + i) % n for i in range(length)]
        idx_set = set(indices)
        
        c1_tri, c2_tri = [], []
        for i in range(n):
            if i in idx_set:
                c1_tri.append(p2.triangles[i])
                c2_tri.append(p1.triangles[i])
            else:
                c1_tri.append(p1.triangles[i])
                c2_tri.append(p2.triangles[i])
                
        return Individual(c1_tri), Individual(c2_tri)