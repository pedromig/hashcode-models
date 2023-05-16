from __future__ import annotations
from typing import TYPE_CHECKING, runtime_checkable, Protocol

if TYPE_CHECKING:
    from .solution import Solution


@runtime_checkable
class Problem(Protocol):
    def __init__(self: Problem) -> None:
        ...

    def empty_solution(self: Problem) -> Solution:
        ...

    def random_solution(self: Problem) -> Solution:
        ...

    def __str__(self: Problem) -> str:
        ...
