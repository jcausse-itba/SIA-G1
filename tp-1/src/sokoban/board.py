"""Sokoban board model.

Provides :class:`SokobanBoard`, an immutable representation of a Sokoban game
state suitable for use with search algorithms (BFS, DFS, Greedy, A*).

The static grid (walls, goals, empty floor) is stored in an
:class:`~sokoban.utils.ImmutableMatrix`, while the mutable entities (player
and boxes) are tracked separately so that state equality and hashing only
depend on the parts that actually change between moves.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Iterator, Self, override

from .utils import Coordinate, Direction, ImmutableMatrix


class Tile(Enum):
    """Static cell types on the Sokoban grid.

    Player and box positions are tracked separately (they move), so the grid
    only records the underlying floor type.
    """

    EMPTY = 0
    WALL = 1
    GOAL = 2


class Board(ABC):
    """Abstract base class for board-game states used with search algorithms."""

    @abstractmethod
    def move(self, direction: Direction) -> Self | None:
        """Apply a move, returning a new state or ``None`` if illegal."""

    @property
    @abstractmethod
    def is_solved(self) -> bool:
        """Whether this state satisfies the goal condition."""

    @abstractmethod
    def is_deadlock(self) -> bool:
        """Whether this state is provably unsolvable."""

    @abstractmethod
    def derived_boards(self) -> Iterator[tuple[Direction, Self]]:
        """Generator that yields tuples of ``(direction, board)`` for each board that can be derived from the current
        board by performing a move in a certain :class:`Direction`."""


class SokobanBoard(Board):
    """Immutable Sokoban game state.

    The board separates *static* terrain (walls, goals, floor) stored in an
    :class:`ImmutableMatrix[Tile]` from *dynamic* entities (player position
    and box positions).  Two boards are equal iff their player position, box
    positions **and** grid are identical; the hash is computed accordingly.

    :param grid: The static terrain matrix.
    :param player_position: Current player :class:`Coordinate`.
    :param box_positions: Iterable of box :class:`Coordinate` values.

    :raise ValueError: If positions are out of bounds, on walls, or duplicated.
    """

    __slots__ = (
        "_grid",
        "_player_position",
        "_box_positions",
        "_goal_coordinates",
        "_deadlock_positions",
        "_hash",
    )

    def __init__(
        self,
        grid: ImmutableMatrix[Tile],
        player_position: Coordinate,
        box_positions: frozenset[Coordinate],
    ) -> None:
        self._grid: ImmutableMatrix[Tile] = grid
        self._player_position: Coordinate = player_position
        self._box_positions: frozenset[Coordinate] = box_positions

        self._goal_coordinates: frozenset[Coordinate] = frozenset(
            grid.find_all(Tile.GOAL)
        )

        self._validate_position(player_position)
        for box in self._box_positions:
            self._validate_position(box)
        if len(self._box_positions) != len(box_positions):
            raise ValueError("Duplicate box positions provided.")

        self._deadlock_positions: frozenset[Coordinate] = self._compute_deadlock_positions()
        self._hash: int = hash((self._player_position, self._box_positions, self._grid))

    @classmethod
    def _create_trusted(
        cls,
        grid: ImmutableMatrix[Tile],
        goal_coordinates: frozenset[Coordinate],
        deadlock_positions: frozenset[Coordinate],
        player_position: Coordinate,
        box_positions: frozenset[Coordinate],
    ) -> Self:
        """Construct a board without re-validating or re-computing static data.

        Used internally by the ``move`` method to avoid redundant work when generating
        successor states during search.
        """
        board = cls.__new__(cls)
        board._grid = grid
        board._player_position = player_position
        board._box_positions = box_positions
        board._goal_coordinates = goal_coordinates
        board._deadlock_positions = deadlock_positions
        board._hash = hash((player_position, box_positions, grid))
        return board

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SokobanBoard):
            return False
        return self._player_position == other._player_position \
            and self._box_positions == other._box_positions \
            and self._grid == other._grid

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        return f"{type(self).__name__}(player={self._player_position!r}, boxes={sorted(self._box_positions)!r}, " + \
                f"grid_shape={self._grid.rows}×{self._grid.cols})"

    def __str__(self) -> str:
        lines: list[str] = []
        for r in range(self._grid.rows):
            row_chars: list[str] = []
            for c in range(self._grid.cols):
                coord = Coordinate(r, c)
                tile = self._grid[coord]
                is_player = coord == self._player_position
                is_box = coord in self._box_positions
                is_goal = tile == Tile.GOAL

                if is_player:
                    row_chars.append("+" if is_goal else "@")
                elif is_box:
                    row_chars.append("*" if is_goal else "$")
                elif tile == Tile.WALL:
                    row_chars.append("#")
                elif is_goal:
                    row_chars.append(".")
                else:
                    row_chars.append(" ")

            lines.append("".join(row_chars))
        return "\n".join(lines)

    @property
    def grid(self) -> ImmutableMatrix[Tile]:
        return self._grid

    @property
    def player_position(self) -> Coordinate:
        return self._player_position

    @property
    def box_positions(self) -> frozenset[Coordinate]:
        return self._box_positions

    @property
    def goal_coordinates(self) -> frozenset[Coordinate]:
        return self._goal_coordinates

    @override
    @property
    def is_solved(self) -> bool:
        return self._box_positions == self._goal_coordinates

    @override
    def move(self, direction: Direction) -> Self | None:
        """Move the player one step in *direction*.

        If the target cell is a wall, returns ``None``.  If a box occupies the
        target cell, the box is pushed one cell further — unless *that* cell is
        a wall or another box, in which case ``None`` is returned.

        :param direction: The direction to move.
        :return: A new :class:`SokobanBoard` with the move applied, or ``None``.
        """
        target = self._player_position + direction.value

        if self._grid[target] == Tile.WALL:
            return None

        # Simple move (no box in the way).
        if target not in self._box_positions:
            return self._create_trusted(
                grid=self._grid,
                goal_coordinates=self._goal_coordinates,
                deadlock_positions=self._deadlock_positions,
                player_position=target,
                box_positions=self._box_positions,
            )

        # Box push — check the cell behind the box.
        box_target = target + direction.value
        if self._grid[box_target] == Tile.WALL or box_target in self._box_positions:
            return None

        new_boxes = (self._box_positions - {target}) | {box_target}
        return self._create_trusted(
            grid=self._grid,
            goal_coordinates=self._goal_coordinates,
            deadlock_positions=self._deadlock_positions,
            player_position=target,
            box_positions=new_boxes,
        )

    @override
    def derived_boards(self) -> Iterator[tuple[Direction, SokobanBoard]]:
        """Yield all reachable successor states with the move that produced them.

        :return: ``(direction, new_board)`` pairs for every legal move.
        """
        for direction in Direction:
            successor = self.move(direction)
            if successor is not None:
                yield direction, successor

    @override
    def is_deadlock(self) -> bool:
        """Check for simple and freeze deadlocks.
        Reference: http://sokobano.de/wiki/index.php?title=How_to_detect_deadlocks

        *Simple* deadlocks are positions from which no sequence of pushes can
        ever move a box to any goal — these are precomputed once per grid.
        *Freeze* deadlocks occur when a box is stuck against walls or other
        stuck boxes on both axes simultaneously.
        """
        return any(
            box not in self._goal_coordinates and (box in self._is_deadlock(box) or self._is_box_frozen(box)) \
            for box in self._box_positions
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_position(self, pos: Coordinate) -> None:
        if not (0 <= pos.row < self._grid.rows and 0 <= pos.col < self._grid.cols):
            raise ValueError(f"Position {pos} is out of grid bounds ({self._grid.rows}×{self._grid.cols}).")
        if self._grid[pos] == Tile.WALL:
            raise ValueError(f"Entity cannot start inside a WALL at {pos}.")

    def _is_wall(self, pos: Coordinate) -> bool:
        return self._grid.get(pos, Tile.WALL) == Tile.WALL

    def _is_box(self, pos: Coordinate) -> bool:
        return pos in self._box_positions

    def _is_deadlock(self, pos: Coordinate) -> bool:
        return pos in self._deadlock_positions

    def _is_box_frozen(self, pos: Coordinate, checking: frozenset[Coordinate] = frozenset()) -> bool:
        """Recursive freeze-deadlock check for a box at *pos*."""
        c = checking | {pos}
        return self._is_axis_blocked(pos, c, True) and self._is_axis_blocked(pos, c, False)

    def _is_axis_blocked(self, pos: Coordinate, checking: frozenset[Coordinate], horizontal: bool) -> bool:
        """Check whether *pos* is blocked along one axis (horizontal or vertical)."""
        neighbours = (Coordinate(pos.row, pos.col - 1), Coordinate(pos.row, pos.col + 1)) if horizontal \
            else (Coordinate(pos.row - 1, pos.col), Coordinate(pos.row + 1, pos.col))

        for neighbour in neighbours:
            if neighbour in checking or self._is_wall(neighbour) or \
                    (self._is_box(neighbour) and self._is_box_frozen(neighbour, checking)):
                return True

        return all([self._is_deadlock(n) for n in neighbours])

    def _compute_deadlock_positions(self) -> frozenset[Coordinate]:
        """Precompute simple deadlock positions via reverse-reachability from goals.

        A cell is a simple deadlock if no sequence of legal pushes (ignoring
        other boxes) can ever move a box from that cell to any goal.
        """
        reachable: set[Coordinate] = set(self._goal_coordinates)
        stack: list[Coordinate] = list(self._goal_coordinates)

        while stack:
            box_pos = stack.pop()
            for direction in Direction:
                dr, dc = direction.value
                # Where the box would have been *before* the push.
                pulled_box = Coordinate(box_pos.row - dr, box_pos.col - dc)
                # Where the puller (player) would have stood.
                puller = Coordinate(box_pos.row - 2 * dr, box_pos.col - 2 * dc)

                if pulled_box in reachable:
                    continue
                if self._is_wall(pulled_box) or self._is_wall(puller):
                    continue

                reachable.add(pulled_box)
                stack.append(pulled_box)

        return frozenset(
            Coordinate(r, c)
            for r in range(self._grid.rows)
            for c in range(self._grid.cols)
            if self._grid[Coordinate(r, c)] != Tile.WALL
            and Coordinate(r, c) not in reachable
        )
