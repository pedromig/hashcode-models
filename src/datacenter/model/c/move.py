from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import *


class LocalMove:
    def __init__(self: LocalMove, problem: Problem,
                 add: tuple[int] | None, remove: tuple[int] | None,
                 swap_segment: bool = False, swap_pool: bool = False) -> None:
        self.problem = problem
        self.add = add
        self.remove = remove
        self.swap_segment = swap_segment
        self.swap_pool = swap_pool

    def __lt__(self: LocalMove, other: LocalMove):
        sadd = self.add if self.add is not None else tuple(),
        sremove = self.remove if self.remove is not None else tuple()
        add = other.add if other.add is not None else tuple(),
        remove = other.remove if other.remove is not None else tuple()
        return (sadd, sremove, self.swap_segment, self.swap_pool) < (add, remove, other.swap_segment, other.swap_pool)

    def __eq__(self: LocalMove, other: LocalMove):
        return self.add == other.add and self.remove == other.remove and self.swap_segment == other.swap_segment and self.swap_pool == other.swap_pool

    def __repr__(self: LocalMove) -> str:
        return self.__str__()

    def __str__(self: LocalMove) -> str:
        return f"({self.add}, {self.remove}, {self.swap_pool}, {self.swap_segment})"
