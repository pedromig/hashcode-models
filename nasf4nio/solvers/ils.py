from __future__ import annotations

from typing import cast, Union, TypeVar, Optional, Protocol, Iterable, Any
from typing_extensions import Self

LocalMove = TypeVar('LocalMove')

T = TypeVar("T", bound=Any) 
    
class TimerProtocol(Protocol):
    def finished(self: Self) -> bool: ...
    
Timer = TypeVar('Timer', bound=TimerProtocol) 
    
class SolutionProtocol(Protocol[T, LocalMove]):
    def objective(self: Self) -> T: ...
    def copy(self: Self) -> Self: ...
    def random_local_moves_wor(self: Self) -> Iterable[LocalMove]: ...
    def objective_increment_local(self: Self, move: LocalMove) -> Optional[LocalMove]: ...
    def step(self: Self, move: LocalMove) -> None: ...
    def perturb(self: Self, ks: int) -> None: ...

Solution = TypeVar('Solution', bound=SolutionProtocol)

class ILS:
    def __init__(self: Self, ks: Optional[int] = 3, zero: Any = 0) -> None:
        self.ks = ks 
        self.zero = zero
  
    def __call__(self: Self, solution: Solution, timer: Timer) -> Solution:
        best = solution.copy()
        bobjv = cast(T, best.objective())
        while not timer.finished():
            for move in solution.random_local_moves_wor():
                incr = cast(T, solution.objective_increment_local(move))
                if incr > self.zero:
                    solution.step(move)
                    break
                if timer.finished():
                    obj = cast(T, solution.objective())
                    if obj > bobjv:
                        return solution
                    else:
                        return best
            else:
                obj = cast(T, solution.objective())
                if obj >= bobjv:
                    best = solution.copy()
                    bobjv = obj
                    print(best.score())
                else:
                    solution = best.copy()
                solution.perturb(self.ks)
        obj = cast(T, solution.objective())
        if obj > bobjv:
            return solution
        else:
            return best