"""Breadth-First Search for Sokoban.

Explores the state space level-by-level, guaranteeing the shortest
solution (in number of moves) is found first.
"""

import time
from collections import deque

from .base import SearchResult
from ..board import Board
from ..node import Node


def _make_result(
    success: bool, solution: list, cost: int,
    expanded_nodes: int, frontier_nodes: int,
    elapsed_seconds: float, operations_done: int,
) -> SearchResult:
    """Build a :class:`SearchResult` with all fields populated."""
    result = SearchResult()
    result.success = success
    result.algorithm = "BFS"
    result.solution = solution
    result.cost = cost
    result.expanded_nodes = expanded_nodes
    result.frontier_nodes = frontier_nodes
    result.elapsed_seconds = elapsed_seconds
    result.operations_done = operations_done
    return result


def bfs(initial_state: Board) -> SearchResult:
    """Run BFS on the given initial board state.

    :param initial_state: The starting Sokoban board.
    :return: A :class:`SearchResult` with the outcome.
    """
    start = time.perf_counter()
    root = Node(initial_state)
    operations = 0

    if root.is_solution():
        return _make_result(
            success=True, solution=[], cost=0,
            expanded_nodes=0, frontier_nodes=1,
            elapsed_seconds=time.perf_counter() - start,
            operations_done=operations,
        )

    frontier: deque[Node] = deque([root])
    frontier_states: set[Board] = {root.board}
    explored: set[Board] = set()
    expanded = 0

    while frontier:
        node = frontier.popleft()
        frontier_states.discard(node.board)
        explored.add(node.board)
        expanded += 1

        for child in node.expand():
            operations += 1
            if child.board in explored or child.board in frontier_states:
                continue
            if child.is_solution():
                return _make_result(
                    success=True, solution=child.path(), cost=child.depth,
                    expanded_nodes=expanded,
                    frontier_nodes=len(frontier) + 1,
                    elapsed_seconds=time.perf_counter() - start,
                    operations_done=operations,
                )
            frontier.append(child)
            frontier_states.add(child.board)

    return _make_result(
        success=False, solution=[], cost=0,
        expanded_nodes=expanded, frontier_nodes=0,
        elapsed_seconds=time.perf_counter() - start,
        operations_done=operations,
    )
