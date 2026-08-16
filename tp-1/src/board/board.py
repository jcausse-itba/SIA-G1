from abc import ABC, abstractmethod
from enum import Enum
from typing import NamedTuple, Self

class Direction(Enum):
    """Cardinal directions the blank (0) tile can move."""

    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

class Board(ABC):
    @abstractmethod
    def move(self, direction: Direction) -> Self | None:
        pass

    @property
    @abstractmethod
    def is_solved(self) -> bool:
        pass

    @abstractmethod
    def has_deadlock(self) -> bool:
        pass