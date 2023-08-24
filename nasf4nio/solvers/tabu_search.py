from __future__ import annotations

import random

from typing import cast, Optional, TypeVar, Protocol, Iterable, Any, List
from typing_extensions import Self

T = TypeVar('T', bound=Any)
LocalMove = TypeVar('LocalMove')

class TimerProtocol(Protocol):
    def finished(self: Self) -> bool: ...
    
Timer = TypeVar('Timer', bound=TimerProtocol)

class SolutionProtocol(Protocol[T, LocalMove]):
    def copy(self: Self) -> Self: ...
    def objective(self: Self) -> T: ...
    def step(self: Self, move: LocalMove) -> None: ...
    def random_local_moves_wor(self: Self) -> Iterable[LocalMove]: ...
    def objective_increment_local(self: Self, move: LocalMove) -> Optional[T]: ...
    def __eq__(self: Self, other: Self) -> bool: ...

Solution = TypeVar('Solution', bound=SolutionProtocol)

class TabuSearch:  
    def __init__(self: Self, length: Optional[int] = 10,
                 zero: Optional[Any] = 0) -> None:
        self.length = length
        self.zero = zero
    
    def __call__(self: Self, solution: Solution, timer: Timer) -> Solution:
        tabu = set()    
        while not timer.finished():
            bincr, best = self.zero, None
            for move in solution.random_local_moves_wor():
                incr = solution.objective_increment_local(move)
                if incr >= bincr:
                    s = solution.copy()
                    s.step(move)
                    if not tabu[s]:
                        best, bincr = s, incr
                if timer.finished():
                    break
            if best is None:
                break
            else:
                tabu.add(best)        
                if len(tabu) > self.length:
                    tabu.pop(0)
        return best