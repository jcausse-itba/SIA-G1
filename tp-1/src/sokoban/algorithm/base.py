from abc import ABC, abstractmethod
from typing import Callable
from ..board import Direction, Board
import psutil

class SearchResult:
    success: bool
    algorithm: str
    solution: list[Direction]
    cost: int
    expanded_nodes: int
    frontier_nodes: int
    elapsed_seconds: float
    operations_done: int

    def __str__(self) -> str:
        status = "ÉXITO" if self.success else "FRACASO"
        out = [
            f"Algoritmo:        {self.algorithm}",
            f"Resultado:        {status}",
            f"Costo solución:   {self.cost if self.success else '-'}",
            f"Nodos expandidos: {self.expanded_nodes}",
            f"Nodos frontera:   {self.frontier_nodes}",
            f"Tiempo (s):       {self.elapsed_seconds:.4f}",
            f"Operaciones:      {self.operations_done}"
        ]
        if self.success:
            out.append("Solución:         " + " ".join(d.name for d in self.solution))
        return "\n".join(out)

class BaseAlgorithm(ABC):
    _process = psutil.Process()

    @classmethod
    def check_memory(cls, limit_gb: float = 4.0) -> bool:
        """Returns True if current process RSS exceeds limit_gb."""
        return cls._process.memory_info().rss > limit_gb * 1024 * 1024 * 1024
    
    # ? May be static
    @abstractmethod
    def search(self, initial_state: Board, heuristic: Callable[[Board], float] = lambda board: 0.0) -> SearchResult:
        pass