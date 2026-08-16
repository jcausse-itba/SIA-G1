from enum import Enum
from typing import NamedTuple, Self, Sequence

class Position(NamedTuple):
    row: int
    col: int

    def move(self, direction: Direction) -> "Position":
        dr, dc = direction.value
        return Position(self.row + dr, self.col + dc)

class Direction(Enum):
    """Cardinal directions the blank (0) tile can move."""

    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

# Player and Box are not here since they would overlap
# ? Maybe add BOX/BOX_ON_GOAL + PLAYER/PLAYER_ON_GOAL, I personally think it's messy
class Tile(Enum):
    EMPTY = 0
    WALL = 1
    GOAL = 2

class SokobanBoard:
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
    ) -> Self:
        board = cls.__new__(cls)
        board.grid = grid
        board.rows = rows
        board.cols = cols
        board.goal_coordinates = goal_coordinates
        board.player_position = player_position
        board.box_positions = box_positions
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
        )

    def _validate_position(self, pos: Position) -> None:
        if not (0 <= pos.row < self.rows and 0 <= pos.col < self.cols):
            raise ValueError(
                f"Position {pos} is out of grid bounds ({self.rows}x{self.cols})."
            )
        if self.grid[pos.row][pos.col] == Tile.WALL:
            raise ValueError(f"Entity cannot start inside a WALL at {pos}.")

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