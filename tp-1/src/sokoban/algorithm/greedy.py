import heapq
import time
from typing import Callable

from .base import BaseAlgorithm, SearchResult
from ..node import Node
from ..board import Board


def _make_result(
    success: bool,
    solution: list,
    cost: int,
    expanded_nodes: int,
    frontier_nodes: int,
    elapsed_seconds: float,
    operations_done: int,
) -> SearchResult:
    """Build a :class:`SearchResult` with all fields populated."""
    result = SearchResult()
    result.success = success
    result.algorithm = "Greedy"
    result.solution = solution
    result.cost = cost
    result.expanded_nodes = expanded_nodes
    result.frontier_nodes = frontier_nodes
    result.elapsed_seconds = elapsed_seconds
    result.operations_done = operations_done
    return result


class PrioritizedNode:
    __slots__ = ("f", "count", "node")

    def __init__(self, f: float, count: int, node: Node) -> None:
        self.f = f
        self.count = count
        self.node = node

    def __lt__(self, other: "PrioritizedNode") -> bool:
        if self.f == other.f:
            return self.count < other.count
        return self.f < other.f


class GreedyBFS(BaseAlgorithm):
    def search(
        self, initial_state: Board, heuristic: Callable[[Board], float] = lambda board: 0.0
    ) -> SearchResult:
        start = time.perf_counter()
        root = Node(initial_state)
        operations = 0
        counter = 0

        if root.is_solution():
            return _make_result(
                success=True,
                solution=[],
                cost=0,
                expanded_nodes=0,
                frontier_nodes=1,
                elapsed_seconds=time.perf_counter() - start,
                operations_done=operations,
            )

        # Priority is based ONLY on the heuristic h(n), ignoring depth (g)
        f_root = heuristic(root.board)
        unexplored: list[PrioritizedNode] = [
            PrioritizedNode(f_root, counter, root)
        ]

        explored: set[Board] = set()
        expanded = 0

        while len(unexplored) > 0:
            if self.check_memory(): break
            p_node = heapq.heappop(unexplored)
            current_node = p_node.node
            current_board = current_node.board

            if current_board in explored:
                continue

            explored.add(current_board)
            expanded += 1

            if current_node.is_solution():
                return _make_result(
                    success=True,
                    solution=current_node.path(),
                    cost=current_node.depth,
                    expanded_nodes=expanded,
                    frontier_nodes=len(unexplored) + 1,
                    elapsed_seconds=time.perf_counter() - start,
                    operations_done=operations,
                )

            for child in current_node.expand():
                operations += 1

                if child.board in explored:
                    continue

                h_cost = heuristic(child.board)
                # f_cost equals only h_cost for Greedy Search
                f_cost = h_cost

                counter += 1
                heapq.heappush(
                    unexplored, PrioritizedNode(f_cost, counter, child)
                )

        return _make_result(
            success=False,
            solution=[],
            cost=0,
            expanded_nodes=expanded,
            frontier_nodes=0,
            elapsed_seconds=time.perf_counter() - start,
            operations_done=operations,
        )