from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .problem import *

class Component:
    def __init__(self: Component, problem: Problem, server: int, pool: int | None, segment: int | None) -> None:
        self.problem = problem
        self.server = server
        self.pool = pool
        self.segment = segment

    def __str__(self: Component) -> str:
        return f"{self.server} {self.pool} {self.segment}"

    def __repr__(self: Component) -> str:
        return self.__str__()
