from pathlib import Path
from board.sokoban import Position, SokobanBoard, Tile

# '#' pared | ' ' piso | '.' objetivo | '$' caja | '*' caja en objetivo
# '@' jugador | '+' jugador en objetivo
_CHAR_TO_TILE = {
    "#": Tile.WALL, " ": Tile.EMPTY, ".": Tile.GOAL,
    "$": Tile.EMPTY, "*": Tile.GOAL, "@": Tile.EMPTY, "+": Tile.GOAL,
}
_BOX_CHARS = {"$", "*"}
_PLAYER_CHARS = {"@", "+"}


def load_level(path: str | Path) -> SokobanBoard:
    """Parsea un nivel Sokoban en notación XSB simplificada.

    :param path: Ruta al archivo de nivel en texto plano.
    :raise ValueError: Si el archivo está vacío o no tiene jugador.
    """
    raw_lines = Path(path).read_text(encoding="utf-8").splitlines()
    lines = [ln for ln in raw_lines if ln != ""] or raw_lines
    if not lines:
        raise ValueError(f"Nivel '{path}' vacío.")

    width = max(len(line) for line in lines)
    padded = [line.ljust(width) for line in lines]

    grid: list[list[Tile]] = []
    player_position: Position | None = None
    box_positions: list[Position] = []

    for r, line in enumerate(padded):
        row: list[Tile] = []
        for c, ch in enumerate(line):
            row.append(_CHAR_TO_TILE.get(ch, Tile.EMPTY))
            if ch in _BOX_CHARS:
                box_positions.append(Position(r, c))
            if ch in _PLAYER_CHARS:
                player_position = Position(r, c)
        grid.append(row)

    if player_position is None:
        raise ValueError(f"Nivel '{path}' no tiene jugador ('@' o '+').")

    return SokobanBoard(grid=grid, player_position=player_position, box_positions=box_positions)