from __future__ import annotations

from typing import cast, TypeVar, Protocol, Optional, Iterable, Any
from typing_extensions import Self

T = TypeVar('T')
LocalMove = TypeVar('LocalMove')

class TimerProtocol(Protocol):
    def finished(self: Self) -> bool: ...
    
Timer = TypeVar('Timer', bound=TimerProtocol) 

class SolutionProtocol(Protocol[T, LocalMove]):
    def step(self: Self, move: LocalMove) -> None: ...
    def random_local_moves_wor(self: Self) -> Iterable[LocalMove]: ...
    def objective_increment_local(self: Self, move: LocalMove) -> Optional[T]: ...

Solution = TypeVar('Solution', bound=SolutionProtocol)
    
class RLS:
    def __init__(self: Self, zero: Optional[Any] = 0):
        self.zero = zero

    def __call__(self: Self, solution: Solution, timer: Timer) -> Solution:
        while not timer.finished():
            for move in solution.random_local_moves_wor():
                delta = cast(T, solution.objective_increment_local(move))
                if delta >= self.zero:
                    solution.step(move)
                    break
                if timer.finished():
                    return solution
            else:
                break
        return solution    

