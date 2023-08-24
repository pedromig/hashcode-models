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
    def local_moves(self: Self) -> Iterable[LocalMove]: ...
    def objective_increment_local(self, move: LocalMove) -> Optional[T]: ...

Solution = TypeVar('Solution', bound=SolutionProtocol)

class BestImprovement: 
    def __init__(self: Self, zero: Optional[Any] = 0) -> None:
        self.zero = zero
    
    def __call__(self: Self, solution: Solution, timer: Timer) -> Solution:
        while timer.finished():
            bincr, bmove = self.zero, None
            for move in solution.local_moves():
                incr = cast(T, solution.objective_increment_local(move))
                if incr >= bincr:
                    bincr = incr
                    bmove = move
                if timer.finished():
                    break
            if bmove is None:
                break
            else:
                solution.step(bmove)
        return solution