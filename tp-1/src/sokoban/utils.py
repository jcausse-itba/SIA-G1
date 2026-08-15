from enum import Enum
from typing import Generic, Iterator, NamedTuple, TypeVar
from math import sqrt


class Direction(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)


class Coordinate(NamedTuple):
    """Immutable ``(row, col)`` position on a 2-D grid.

    Extends :class:`~typing.NamedTuple`, so instances are lightweight, hashable,
    and support tuple unpacking::

        pos = Coordinate(2, 5)
        row, col = pos           # unpacking
        {pos}                    # usable in sets / dict keys
        pos + Direction.UP.value # arithmetic with direction deltas

    :param row: Row index (0-indexed, grows downward).
    :param col: Column index (0-indexed, grows rightward).
    """

    row: int
    col: int

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Coordinate):
            return False
        return self.row == other.row and self.col == other.col

    def __hash__(self) -> int:
        return hash((self.row, self.col))

    def __add__(self, other: object) -> Coordinate:
        if not isinstance(other, tuple) or len(other) != 2:
            return NotImplemented
        return Coordinate(self.row + other[0], self.col + other[1])

    def __sub__(self, other: object) -> Coordinate:
        if not isinstance(other, tuple) or len(other) != 2:
            return NotImplemented
        return Coordinate(self.row - other[0], self.col - other[1])

    def __repr__(self) -> str:
        return f"Coordinate(row={self.row}, col={self.col})"

    def __str__(self) -> str:
        return f"({self.row}, {self.col})"

    def distance(self, other: Coordinate) -> float:
        """Calculate the usual (d2) distance between this coordinate and another :class:`Coordinate`."""
        return sqrt((self.row - other.row) ** 2 + (self.col - other.col) ** 2)

    def manhattan_distance(self, other: Coordinate) -> int:
        """Compute the Manhattan (d1) distance to *other*.

        :param other: Target coordinate.
        :return: ``|Δrow| + |Δcol|``.
        """
        return abs(self.row - other.row) + abs(self.col - other.col)


_T = TypeVar("_T")


class ImmutableMatrix(Generic[_T]):
    """Immutable, tuple-backed 2-D matrix.

    Every row is a ``tuple[_T, ...]`` and the matrix itself is a
    ``tuple[tuple[_T, ...], ...]``, so the entire structure is deeply
    immutable and hashable.

    Use :class:`ImmutableMatrixBuilder` to construct instances
    incrementally.

    :param matrix: A tuple of equal-length row tuples.
    :raise TypeError:  If *matrix* is not a tuple of tuples.
    :raise ValueError: If rows have inconsistent lengths.
    """

    __slots__ = ("_rows", "_cols", "_matrix", "_hash")

    def __init__(self, matrix: tuple[tuple[_T, ...], ...]) -> None:
        if not isinstance(matrix, tuple) or not all(isinstance(r, tuple) for r in matrix):
            raise TypeError("matrix must be a tuple of tuples")

        if len(matrix) == 0:
            self._rows: int = 0
            self._cols: int = 0
        else:
            col_count = len(matrix[0])
            if any(len(r) != col_count for r in matrix):
                raise ValueError("All rows must have the same length")
            self._rows = len(matrix)
            self._cols = col_count

        self._matrix: tuple[tuple[_T, ...], ...] = matrix
        self._hash: int = hash(matrix)

    def __getitem__(self, position: Coordinate | tuple[int, int]) -> _T:
        """Return the element at ``(row, col)``.

        :param position: ``(row, col)`` pair or :class:`Coordinate`.
        :raise IndexError: If the position is out of bounds.
        """
        row, col = position
        if not (0 <= row < self._rows and 0 <= col < self._cols):
            raise IndexError(
                f"position ({row}, {col}) out of range for a "
                f"{self._rows}×{self._cols} matrix"
            )
        return self._matrix[row][col]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImmutableMatrix):
            return False
        return self._matrix == other._matrix

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._matrix!r})"

    def __str__(self) -> str:
        if self._rows == 0:
            return "(empty)"
        col_widths = [
            max(len(str(self._matrix[r][c])) for r in range(self._rows))
            for c in range(self._cols)
        ]
        lines: list[str] = []
        for row in self._matrix:
            cells = [str(val).rjust(w) for val, w in zip(row, col_widths)]
            lines.append(" ".join(cells))
        return "\n".join(lines)

    def __contains__(self, value: object) -> bool:
        """Check whether *value* exists anywhere in the matrix."""
        return any(value in row for row in self._matrix)

    def __iter__(self) -> Iterator[tuple[_T, ...]]:
        """Iterate over rows (each row is a tuple)."""
        return iter(self._matrix)

    def __len__(self) -> int:
        """Total number of cells (``rows × cols``)."""
        return self._rows * self._cols

    @property
    def rows(self) -> int:
        """Number of rows."""
        return self._rows

    @property
    def cols(self) -> int:
        """Number of columns."""
        return self._cols

    def get(self, position: Coordinate | tuple[int, int], default: _T | None = None) -> _T | None:
        """Return the element at *position*, or *default* if out of bounds.

        :param position: ``(row, col)`` pair.
        :param default: Value to return when *position* is outside the matrix.
        """
        row, col = position
        if 0 <= row < self._rows and 0 <= col < self._cols:
            return self._matrix[row][col]
        return default

    def find(self, value: _T) -> Coordinate | None:
        """Return the :class:`Coordinate` of the first occurrence of *value*, or ``None``.

        Scans in row-major order.
        """
        for r, row in enumerate(self._matrix):
            for c, cell in enumerate(row):
                if cell == value:
                    return Coordinate(r, c)
        return None

    def find_all(self, value: _T) -> list[Coordinate]:
        """Return a list of :class:`Coordinate` for every occurrence of *value*."""
        return [
            Coordinate(r, c)
            for r, row in enumerate(self._matrix)
            for c, cell in enumerate(row)
            if cell == value
        ]


class ImmutableMatrixBuilder(Generic[_T]):
    """Builder for :class:`ImmutableMatrix` instances.

    Accumulates ``(Coordinate, value)`` pairs into a mutable list-of-lists,
    automatically expanding to keep the matrix rectangular.  Unset cells are
    filled with *default* when :meth:`build` is called.

    :param default: The fill value for cells that are never explicitly set.

    Example::

        builder = ImmutableMatrixBuilder(default=0)
        builder.add(Coordinate(0, 5), 1)
        builder.add(Coordinate(2, 3), 2)
        matrix = builder.build()   # 3 rows × 6 cols, gaps filled with 0
    """

    __slots__ = ("_data", "_default", "_max_row", "_max_col")

    def __init__(self, default: _T) -> None:
        self._data: dict[tuple[int, int], _T] = {}
        self._default: _T = default
        self._max_row: int = -1
        self._max_col: int = -1

    def add(self, coordinate: Coordinate | tuple[int, int], value: _T) -> ImmutableMatrixBuilder[_T]:
        """Set the cell at *coordinate* to *value*.

        If the coordinate extends beyond the current bounds, the matrix
        dimensions are expanded accordingly.  The builder is returned so
        calls can be chained.

        :param coordinate: ``(row, col)`` target position.
        :param value: The value to store.
        :raise ValueError: If row or col is negative.
        :return: ``self`` (for chaining).
        """
        row, col = coordinate
        if row < 0 or col < 0:
            raise ValueError(f"Coordinate must be non-negative, got ({row}, {col})")
        self._data[(row, col)] = value
        if row > self._max_row:
            self._max_row = row
        if col > self._max_col:
            self._max_col = col
        return self

    def build(self) -> ImmutableMatrix[_T]:
        """Freeze the accumulated data into an :class:`ImmutableMatrix`.

        The resulting matrix has dimensions ``(max_row + 1) × (max_col + 1)``.
        Every cell not explicitly set via :meth:`add` is filled with the
        *default* value provided at construction time.

        :raise ValueError: If no elements have been added.
        :return: A new :class:`ImmutableMatrix`.
        """
        if self._max_row < 0 or self._max_col < 0:
            raise ValueError("Cannot build an empty matrix; add at least one element")

        num_rows = self._max_row + 1
        num_cols = self._max_col + 1

        rows: list[list[_T]] = [
            [self._default] * num_cols for _ in range(num_rows)
        ]

        for (r, c), value in self._data.items():
            rows[r][c] = value

        return ImmutableMatrix(tuple(tuple(row) for row in rows))
