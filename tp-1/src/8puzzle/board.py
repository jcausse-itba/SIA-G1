"""8-Puzzle board model.

Provides :class:`EightPuzzleBoard`, an immutable representation of an 8-puzzle
game state suitable for use with search algorithms.
"""

from enum import Enum
from typing import Iterator


_SIZE = 3


class Direction(Enum):
    """Cardinal directions the blank (0) tile can move."""

    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)


class EightPuzzleBoard:
    """Immutable representation of an 8-puzzle board state.

    The board is a 3×3 grid stored internally as a length-9 tuple in
    **row-major** order. The blank space is represented by the value ``0``.

    The class is designed as a *value object*: two boards with the same tile
    arrangement (even if rotated) are considered equal, and instances are
    hashable so they can be stored in sets or used as dictionary keys (useful for
    visited-state tracking in search algorithms).

    :param tiles: ``tuple[int, ...] | list[int]``: a length-9 tuple representing the board in
    **row-major** order. The blank space is represented by the value ``0``.

    :raise TypeError: If *tiles* is not a tuple or list.
    :raise ValueError: If *tiles* does not contain exactly the integers ``0``–``8``
    (including cases where an integer is repeated).
    """

    __slots__ = ("_tiles", "_blank_index", "_hash")

    ####################
    ### CONSTRUCTORS ###
    ####################

    def __init__(self, tiles: tuple[int, ...] | list[int]) -> None:
        if not (isinstance(tiles, list) or isinstance(tiles, tuple)):
            raise TypeError("Argument must be a list or tuple")

        expected = set(range(_SIZE ** 2))
        if len(tiles) != _SIZE ** 2 or set(tiles) != expected:
            raise ValueError(f"Argument must be a permutation of {sorted(expected)}, got {tiles!r}")

        self._tiles: tuple[int, ...] = tuple(tiles)
        self._blank_index: int = tiles.index(0)
        self._hash: int = hash(self._tiles)

    @staticmethod
    def from_matrix(m) -> EightPuzzleBoard:
        """Matrix-based constructor.

        :param m: 3x3 matrix using lists or tuples.

        :raise TypeError: If ``m`` is not a 3x3 matrix.
        :raise ValueError: If ``m`` does not contain exactly the integers ``0`` - ``8``
        """
        err_msg = "Argument must be a 3x3 integer matrix."

        if not (isinstance(m, list) or isinstance(m, tuple)) or len(m) != _SIZE:
            raise TypeError(err_msg)

        for e in m:
            if not (isinstance(e, list) or isinstance(e, tuple)) or len(e) != _SIZE:
                raise TypeError(err_msg)

        try:
            return EightPuzzleBoard(tuple(m[0]) + tuple(m[1]) + tuple(m[2]))
        except ValueError:
            raise ValueError(err_msg)

    #############################
    ### USEFUL DUNDER METHODS ###
    #############################

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EightPuzzleBoard):
            return False
        return self._tiles == other._tiles

    def __hash__(self) -> int:
        return self._hash

    def __getitem__(self, position: tuple[int, int]) -> int:
        row, col = position
        if not (0 <= row < _SIZE and 0 <= col < _SIZE):
            raise IndexError(f"position {position} out of range for a {_SIZE}×{_SIZE} board")
        return self._tiles[row * _SIZE + col]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._tiles!r})"

    def __str__(self) -> str:
        s = ""
        for i in range(_SIZE):
            row = self._tiles[i * _SIZE : (i + 1) * _SIZE]
            s += " ".join([str(x) for x in row]) + "\n"
        return s

    ##################
    ### PROPERTIES ###
    ##################

    @property
    def tiles(self) -> tuple[int, ...]:
        """Get the tiles represented by this :class:`EightPuzzleBoard`."""
        return self._tiles

    @property
    def blank_position(self) -> tuple[int, int]:
        """Get the position of the blank tiles represented by this :class:`EightPuzzleBoard`."""
        return divmod(self._blank_index, _SIZE)

    ###############
    ### METHODS ###
    ###############

    def move(self, direction: Direction) -> EightPuzzleBoard | None:
        """Slide a tile into the blank space from *direction*.

        This effectively moves the blank one step in the given :class:`Direction`.
        Returns a new board with the move applied, or ``None`` if the move is illegal (would go out of bounds).

        :param direction: :class:`Direction`: The direction to move the blank.
        """
        blank_row, blank_col = self.blank_position
        d_row, d_col = direction.value
        new_row, new_col = blank_row + d_row, blank_col + d_col

        if not (0 <= new_row < _SIZE and 0 <= new_col < _SIZE):
            return None

        new_index = new_row * _SIZE + new_col
        tiles = list(self._tiles)
        tiles[self._blank_index], tiles[new_index] = tiles[new_index], tiles[self._blank_index]
        return EightPuzzleBoard(tiles)

    def neighbours(self) -> Iterator[tuple[Direction, EightPuzzleBoard]]:
        """Yield all reachable neighbour states with the move that produced them. Each time, returns a pair that
        contains a Direction and the board generated by moving in that direction.
        """
        for direction in Direction:
            neighbour = self.move(direction)
            if neighbour is not None:
                yield direction, neighbour

    """
    If you, like me, think not every 8-puzzle board is solvable, you are actually right.
    See: https://www.geeksforgeeks.org/dsa/check-instance-8-puzzle-solvable/
    
    This could be implemented like this:
    
    def is_solvable(self) -> bool:
        values = [t for t in self._tiles if t != 0]
        inversions = sum(1
            for i in range(len(values))
                for j in range(i + 1, len(values))
                    if values[i] > values[j]
        )
       return inversions % 2 == 0

    This was fun to code, but it is rendered useless by the fact that every rotation or reflection of the canonical
    solution:
          1 2 3
          4 5 6
          7 8 _
    is accepted as a valid solution. In this scenario, every board is solvable. Why? This is why:
    
    The 8-puzzle state space splits into exactly two disconnected components based on the parity of the tile 
    permutation (even vs. odd number of inversions). One can only move between states within the same parity class.
    The canonical goal (1,2,3,4,5,6,7,8,0) has 0 inversions, so it belongs to the even class. Therefore, only 
    even-parity states can reach it. But some transforms of the goal land in the other class:
    
    Transform   | Tiles (ignoring 0) | Inversions | Parity
    ------------|--------------------|------------|--------
    Identity    | 1,2,3,4,5,6,7,8    | 0          | even
    90° CW      | 7,4,1,8,5,2,6,3    | 15         | odd
    180°        | 8,7,6,5,4,3,2,1    | 28         | even
    270° CW     | 3,6,2,5,8,1,4,7    | 12         | even
    Flip horiz. | 3,2,1,6,5,4,8,7    | 7          | odd
    Flip vert.  | 7,8,4,5,6,1,2,3    | 16         | even
    Main diag.  | 1,4,7,2,5,8,3,6    | 9          | odd
    Anti-diag.  | 6,3,8,5,2,7,4,1    | 18         | even
    
    Since both parities are represented among the goals, every initial state — even or odd — can reach at least one 
    of them. Then, the is_solvable() check becomes trivially True always.
    
    Just in case, for the sake of clarification, this was human-written, not AI-written.
    """
