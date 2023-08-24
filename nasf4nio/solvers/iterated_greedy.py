from __future__ import annotations

import random

from typing import TypeVar, Protocol, Optional, Iterable, TypedDict, cast
from typing_extensions import Self, Unpack

from operator import itemgetter 

Component = TypeVar("Component")

class Comparable(Protocol): 
    def __lt__(self: Self, other: Self) -> bool: ...

T = TypeVar("T", bound=Comparable, covariant=True)

class TimerProtocol(Protocol):
    def finished(self: Self) -> bool: ...
    
Timer = TypeVar('Timer', bound=TimerProtocol)

class SolutionProtocol(Protocol[T, Component]):
    def copy(self: Self) -> Self: ...
    def feasible(self: Self) -> bool: ... 
    def objective(self: Self) -> Optional[T]: ...
    def upper_bound(self: Self)-> Optional[T]: ... 
    def add_moves(self: Self) -> Iterable[Component]: ...
    def upper_bound_increment_add(self: Self, component: Component) -> Optional[T]: ...
    def random_remove_move(self: Self) -> Optional[Component]: ...
    def add(self: Self, component: Component) -> None: ...
    def remove(self: Self, component: Component) -> None: ...

Solution = TypeVar('Solution', bound=SolutionProtocol) 

class IteratedGreedy:
    def __init__(self: Self, alpha: Optional[float] = 0.1, 
                 ks: Optional[int] = 0) -> None:
        self.alpha = alpha  
        self.ks = ks
 
        self.__filter = self.__threshold if alpha else self.__no_threshold
  
    def __call__(self: Self, solution: Solution, timer: Timer) -> Solution: 
        best, bobjv = (solution.copy(), solution.objective()) if solution.feasible() else (None, None)
        while not timer.finished():
            candidates = [(solution.upper_bound_increment_add(c), c) for c in solution.add_moves()]
            while len(candidates) != 0:
                c = self.__filter(candidates)
                solution.add(c)
                if bobjv is not None and cast(T, solution.objective()) < bobjv:
                    break
                candidates = [(solution.upper_bound_increment_add(c), c) for c in solution.add_moves()] 

            if solution.feasible():
                obj = cast(T, solution.objective())
                if bobjv is None or obj > bobjv: 
                    best, bobjv = solution.copy(), obj

            for _ in range(self.ks):
                c = solution.random_remove_move()
                if c is not None:
                    solution.remove(c)
        return best
     
    def __threshold(self: Self, candidates: list[tuple[T, Component]]) -> Component: 
        cmin = min(candidates, key=itemgetter(0))[0]
        cmax = max(candidates, key=itemgetter(0))[0]
        thresh = cmin + self.alpha * (cmax - cmin)
        rcl = [c for decr, c in candidates if decr <= thresh]
        return random.choice(rcl)
        
    def __no_threshold(self: Self, candidates: list[tuple[T, Component]]) -> Component:
        cmax = max(candidates, key=itemgetter(0))[0]
        rcl = [c for decr, c in candidates if decr == cmax]
        return random.choice(rcl) 