from __future__ import annotations

import random

from typing import TypeVar, Protocol, Optional, Iterable, List, Tuple, cast
from typing_extensions import Self

from operator import itemgetter

class ComparableAndAddable(Protocol): 
    def __lt__(self: Self, other: Self) -> bool: ...
    def __add__(self: Self, other: Self) -> Self: ...
        
T = TypeVar("T", bound=ComparableAndAddable, covariant=True)

Component = TypeVar("Component")

class Comparable(Protocol): 
    def __lt__(self: Self, other: Self) -> bool: ...
        
class TimerProtocol(Protocol):
    def finished(self: Self) -> bool: ...
    
Timer = TypeVar('Timer', bound=TimerProtocol) 

class SolutionProtocol(Protocol[T, Component]):
    def copy(self: Self) -> Self: ...
    def objective(self: Self) -> Optional[T]: ...
    def upper_bound(self: Self) -> Optional[T]: ...
    def feasible(self) -> bool: ...
    def add_moves(self) -> Iterable[Component]: ...
    def upper_bound_increment_add(self, component: Component) -> Optional[T]: ...
    def add(self, component: Component) -> None: ...
    
Solution = TypeVar("Solution", bound=SolutionProtocol)  
BeamList = List[Tuple[T, Solution]]

class BeamSearch:
    def __init__(self: Self, bw: Optional[int] = 10) -> None:
        self.bw = bw 
    
    def __call__(self: Self, solution: Solution, timer: Timer) -> Solution:
        best, bobjv = ((solution, solution.objective() ) if solution.feasible() else (None, None))
        beam: BeamList = [(solution.upper_bound(), solution)]
        while not timer.finished():
            candidates = []
            for ub, s in beam:
                for c in s.add_moves():
                    candidates.append((ub + cast(T, s.upper_bound_increment_add(c)), s, c))
            if not len(candidates):
                break
            candidates.sort(reverse=True, key=itemgetter(0))
            beam: BeamList = []
            for ub, s, c in candidates[:self.bw]:
                s: Solution = s.copy()
                s.add(c)
                if s.feasible():
                    obj = cast(T, s.objective())
                    if bobjv is None or obj > bobjv:
                        best, bobjv = s, obj
                beam.append((ub, s))
        return best
 