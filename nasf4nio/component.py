from __future__ import annotations
from typing import TYPE_CHECKING, runtime_checkable, Protocol

if TYPE_CHECKING:
    from .problem import Problem

@runtime_checkable
class Component(Protocol):
    def __init__(self: Component, problem: Problem) -> None: ...

    def __str__(self: Component) -> str: ...
