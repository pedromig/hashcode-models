from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import *


class Component:
    def __init__(self: Component, problem: Problem, server: int,
                 pool: int | None = None, segment: int | None = None) -> None:
        self.problem = problem
        self.server = server
        self.pool = pool
        self.segment = segment

    def __lt__(self: Component, other: Component) -> bool:
        return (self.server, self.pool, self.segment) < (other.server, other.pool, other.segment)

    def __eq__(self: Component, other: Component) -> bool:
        return self.server == other.server and self.pool == other.pool and self.segment == other.segment

    def __repr__(self: Component) -> str:
        return self.__str__()

    def __str__(self: Component) -> str:
        row, _, __ = self.problem.segments[self.segment]
        return f"({self.server}, {self.pool}, {row}, {self.segment})"
