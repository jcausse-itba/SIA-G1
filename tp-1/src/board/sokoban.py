from enum import Enum
from typing import NamedTuple, Self, Sequence
from board import Direction, Board

class Position(NamedTuple):
    row: int
    col: int

    def move(self, direction: Direction) -> "Position":
        dr, dc = direction.value
        return Position(self.row + dr, self.col + dc)

# Player and Box are not here since they would overlap
# ? Maybe add BOX/BOX_ON_GOAL + PLAYER/PLAYER_ON_GOAL, I personally think it's messy
class Tile(Enum):
    EMPTY = 0
    WALL = 1
    GOAL = 2

class SokobanBoard(Board):
    # ? I understand that Sequence is Python's equivalent of Collection in Java
    def __init__(
            self, 
            grid: Sequence[Sequence[Tile]],
            player_position: Position,
            box_positions: Sequence[Position]
            ) -> None:
        if not grid or not grid[0]:
            raise ValueError("Board grid must be a non-empty 2D structure.")

        self.rows: int = len(grid)
        self.cols: int = len(grid[0])

        if any(len(row) != self.cols for row in grid):
            raise ValueError("All rows in the grid must have the same length.")

        self.grid: tuple[tuple[Tile,...],...] = tuple(
            tuple(row) for row in grid
        )

        self.player_position: Position = player_position
        self.box_positions: frozenset[Position] = frozenset(box_positions)

        # ? May not be necessary, but faster checks
        self.goal_coordinates: frozenset[Position] = frozenset(
            Position(r,c)
            for r, row in enumerate(self.grid)
            for c, tile in enumerate(row)
            if tile == Tile.GOAL
        )

        self._validate_position(self.player_position)

        for box_pos in self.box_positions:
            self._validate_position(box_pos)

        if len(self.box_positions) != len(box_positions):
            raise ValueError("Duplicate box positions provided.")

        self.deadlock_positions: frozenset[Position] = self._compute_deadlock_positions()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SokobanBoard):
            return NotImplemented
        return (
            self.player_position == other.player_position
            and self.box_positions == other.box_positions
            and self.grid == other.grid
        )

    def __hash__(self) -> int:
        return hash((self.player_position, self.box_positions, self.grid))

    @property
    def is_solved(self) -> bool:
        return self.box_positions == self.goal_coordinates

    @classmethod
    def _create_trusted(
        cls,
        grid: tuple[tuple[Tile, ...], ...],
        rows: int,
        cols: int,
        goal_coordinates: frozenset[Position],
        player_position: Position,
        box_positions: frozenset[Position],
        deadlock_positions: frozenset[Position]
    ) -> Self:
        board = cls.__new__(cls)
        board.grid = grid
        board.rows = rows
        board.cols = cols
        board.goal_coordinates = goal_coordinates
        board.player_position = player_position
        board.box_positions = box_positions
        board.deadlock_positions = deadlock_positions
        return board

    def move(self, direction: Direction) -> Self | None:
        target_pos = self.player_position.move(direction)

        if self.grid[target_pos.row][target_pos.col] == Tile.WALL:
            return None

        if target_pos not in self.box_positions:
            return self._create_trusted(
                grid=self.grid,
                rows=self.rows,
                cols=self.cols,
                goal_coordinates=self.goal_coordinates,
                player_position=target_pos,
                box_positions=self.box_positions,
                deadlock_positions=self.deadlock_positions
            )

        box_target = target_pos.move(direction)
        if (
            self.grid[box_target.row][box_target.col] == Tile.WALL
            or box_target in self.box_positions
        ):
            return None

        new_boxes = (self.box_positions - {target_pos}) | {box_target}
        return self._create_trusted(
            grid=self.grid,
            rows=self.rows,
            cols=self.cols,
            goal_coordinates=self.goal_coordinates,
            player_position=target_pos,
            box_positions=new_boxes,
            deadlock_positions=self.deadlock_positions
        )

    def _validate_position(self, pos: Position) -> None:
        if not (0 <= pos.row < self.rows and 0 <= pos.col < self.cols):
            raise ValueError(
                f"Position {pos} is out of grid bounds ({self.rows}x{self.cols})."
            )
        if self.grid[pos.row][pos.col] == Tile.WALL:
            raise ValueError(f"Entity cannot start inside a WALL at {pos}.")

    def _is_wall(self, pos: Position) -> bool:
        if not (0 <= pos.row < self.rows and 0 <= pos.col < self.cols):
            return True
        return self.grid[pos.row][pos.col] == Tile.WALL


    # DEADLOCKS http://sokobano.de/wiki/index.php?title=How_to_detect_deadlocks

    # Basically there are 3 types of deadlocks: Simple, Freeze and Corral
    # Simple can be computed without boxes, so they should be added as an attribute
    # Freeze are box-dependent, so they should be calculated per call
    # ? According to the wiki, Corral is too complex, maybe we should omit them, if so EXPLAIN IN THE PRESENTATION
    # All of these implementations are taken from the wiki
    def has_deadlock(self) -> bool:
        for box_pos in self.box_positions:
            if box_pos in self.goal_coordinates:
                continue

            if box_pos in self.deadlock_positions:
                return True

            if self._is_box_frozen(box_pos):
                return True

        return False

    def _is_box_frozen(self, pos: Position, checking: frozenset[Position] = frozenset()) -> bool:
        checking = checking | {pos}
        return (
            self._is_axis_blocked(pos, checking=checking, horizontal=True) 
            and self._is_axis_blocked(pos, checking=checking)
        )

    def _is_axis_blocked(self, pos: Position, checking: frozenset[Position], horizontal: bool = False) -> bool:
        if horizontal:
            neighbor_a = Position(pos.row, pos.col - 1)
            neighbor_b = Position(pos.row, pos.col + 1)
        else:
            neighbor_a = Position(pos.row - 1, pos.col)
            neighbor_b = Position(pos.row + 1, pos.col)

        for neighbor in (neighbor_a, neighbor_b):
            # If there as wall/box on either side
            if (neighbor in checking or self._is_wall(neighbor)) or (neighbor in self.box_positions and self._is_box_frozen(neighbor, checking)):
                return True

        if neighbor_a in self.deadlock_positions and neighbor_b in self.deadlock_positions:
            return True

        return False


    def _compute_deadlock_positions(self) -> frozenset[Position]:
        reachable: set[Position] = set(self.goal_coordinates)
        stack: list[Position] = list(self.goal_coordinates)

        while stack:
            box_pos = stack.pop()
            for direction in Direction:
                dr, dc = direction.value
                pulled_box_pos = Position(box_pos.row - dr, box_pos.col - dc)
                puller_pos = Position(box_pos.row - 2 * dr, box_pos.col - 2 * dc)

                if pulled_box_pos in reachable:
                    continue

                if self._is_wall(pulled_box_pos) or self._is_wall(puller_pos):
                    continue

                reachable.add(pulled_box_pos)
                stack.append(pulled_box_pos)

        return frozenset(
            Position(r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.grid[r][c] != Tile.WALL and Position(r, c) not in reachable
        )
