from .board import Board
from .utils import Direction
from typing import Optional, Iterator, Callable


class Node:

    __slots__ = ("_board", "_depth", "_parent", "_direction")

    def __init__(self,
        board: Board,
        depth: int = 0,
        parent: Optional[Node] = None,
        direction: Optional[Direction] = None
    ) -> None:
        self._board: Board = board
        self._depth: int = depth
        self._parent: Optional[Node] = parent
        self._direction: Optional[Direction] = direction

    @property
    def board(self) -> Board:
        return self._board

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def parent(self) -> Optional[Node]:
        return self._parent

    @property
    def direction(self) -> Optional[Direction]:
        return self._direction

    def expand(self) -> Iterator[Node]:
        for direction, board in self._board.derived_boards():
            if not board.is_deadlock():
                yield Node(board, self._depth + 1, self, direction)

    def is_solution(self) -> bool:
        return self._board.is_solved

    def path(self) -> list[Direction]:
        """Get the path of actions (directions moved) needed to get to the current node"""
        ret: list[Direction] = []
        node: Node | None = self
        while node is not None:
            if node._direction is not None:
                ret.insert(0, node._direction)
            node = node.parent
        return ret
