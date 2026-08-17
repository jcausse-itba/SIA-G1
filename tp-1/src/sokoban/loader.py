"""Sokoban level loader.

Parses level files in the standard XSB text format into
:class:`~sokoban.map.SokobanBoard` instances.

XSB character reference::

    #  wall
       empty floor
    .  goal
    $  box
    *  box on goal
    @  player
    +  player on goal
"""

from pathlib import Path

from .board import SokobanBoard, Tile
from .utils import Coordinate, ImmutableMatrixBuilder


_CHAR_TO_TILE: dict[str, Tile] = {
    "#": Tile.WALL,
    " ": Tile.EMPTY,
    ".": Tile.GOAL,
    "$": Tile.EMPTY,   # box sits on empty floor
    "*": Tile.GOAL,    # box sits on a goal
    "@": Tile.EMPTY,   # player sits on empty floor
    "+": Tile.GOAL,    # player sits on a goal
}

_BOX_CHARS: set[str] = {"$", "*"}
_PLAYER_CHARS: set[str] = {"@", "+"}


def load_level(path: str | Path) -> SokobanBoard:
    """Parse a Sokoban level file in XSB notation.

    :param path: Path to the level file (plain text).
    :raise ValueError: If the file is empty or contains no player character.
    :return: A fully constructed :class:`SokobanBoard`.
    """
    raw_lines = Path(path).read_text(encoding="utf-8").splitlines()
    lines = [ln for ln in raw_lines if ln != ""] or raw_lines
    if not lines:
        raise ValueError(f"Level '{path}' is empty.")

    # Pad all lines to the same width so the grid is rectangular.
    width = max(len(line) for line in lines)
    padded = [line.ljust(width) for line in lines]

    builder: ImmutableMatrixBuilder[Tile] = ImmutableMatrixBuilder(default=Tile.EMPTY)
    player_position: Coordinate | None = None
    box_positions: list[Coordinate] = []

    for r, line in enumerate(padded):
        for c, ch in enumerate(line):
            tile = _CHAR_TO_TILE.get(ch, Tile.EMPTY)
            builder.add(Coordinate(r, c), tile)

            if ch in _BOX_CHARS:
                box_positions.append(Coordinate(r, c))
            if ch in _PLAYER_CHARS:
                player_position = Coordinate(r, c)

    if player_position is None:
        raise ValueError(f"Level '{path}' has no player ('@' or '+').")

    grid = builder.build()
    return SokobanBoard(
        grid=grid,
        player_position=player_position,
        box_positions=frozenset(box_positions),
    )
