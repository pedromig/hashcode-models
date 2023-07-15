from __future__ import annotations
from typing import TYPE_CHECKING, runtime_checkable, Protocol, TypeVar

if TYPE_CHECKING:
    from .problem import Problem

@runtime_checkable
class LocalMove(Protocol):
    def __init__(self: LocalMove, problem: Problem) -> None: ...

    def __str__(self: LocalMove) -> str: ...
