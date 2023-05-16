from __future__ import annotations
from typing import TYPE_CHECKING, runtime_checkable, Protocol, Iterable, TypeVar

if TYPE_CHECKING:
    from .move import LocalMove
    from .component import Component
    from .problem import Problem

T = TypeVar("T", bound=float)


@runtime_checkable
class Solution(Protocol):
    def __init__(self: Solution, problem: Problem) -> None:
        ...

    def copy(self: Solution) -> Solution:
        ...

    def feasible(self: Solution) -> bool:
        ...

    def enum_add_move(self: Solution) -> Iterable[Component]:
        ...

    def enum_heuristic_add_move(self: Solution) -> Iterable[Component]:
        ...

    def enum_remove_move(self: Solution) -> Iterable[Component]:
        ...

    def enum_local_move(self: Solution) -> Iterable[LocalMove]:
        ...

    def enum_random_local_move_wor(self: Solution) -> Iterable[LocalMove]:
        ...

    def random_add_move(self: Solution) -> Component:
        ...

    def random_remove_move(self: Solution) -> Component:
        ...

    def random_local_move(self: Solution) -> LocalMove:
        ...

    def add(self: Solution, component: LocalMove) -> None:
        ...

    def remove(self: Solution, component: LocalMove) -> None:
        ...

    def step(self: Solution, move: LocalMove) -> None:
        ...

    def perturb(self: Solution) -> None:
        ...

    def score(self: Solution) -> T:
        ...

    def objective_value(self: Solution) -> T:
        ...

    def upper_bound(self: Solution) -> T:
        ...

    def objective_increment_local(self: Solution, move: LocalMove) -> T:
        ...

    def objective_increment_add(self: Solution, component: Component) -> T:
        ...

    def objective_increment_remove(self: Solution, component: Component) -> T:
        ...

    def upper_bound_increment_add(self: Solution, component: Component) -> T:
        ...

    def upper_bound_increment_remove(self: Solution, component: Component) -> T:
        ...

    def __str__(self: Solution) -> str:
        ...
