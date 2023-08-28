from __future__ import annotations

import random
import logging

from typing import TypeVar, Protocol, Optional, Iterable, TypedDict, cast
from typing_extensions import Self, Unpack

from itertools import islice
from operator import itemgetter

Component = TypeVar("Component")

class Comparable(Protocol): 
    def __lt__(self: Self, other: Self) -> bool: ...

T = TypeVar("T", bound=Comparable, covariant=True)

class TimerProtocol(Protocol):
    def finished(self: Self) -> bool: ...
    
Timer = TypeVar('Timer', bound=TimerProtocol)

class HeuristicConstruction: 
    def __call__(self: Self, solution: Solution) -> Solution:
        while (c := solution.heuristic_add_move()) is not None:
            solution.add(c)
        return solution

    class SolutionProtocol(Protocol[Component]):
        def heuristic_add_move(self: Self) -> Optional[Component]: ...
        def add(self: Self, component: Component) -> None: ...  
        
    Solution = TypeVar("Solution", bound=SolutionProtocol)
    
    
class NarrowGuidedHeuristicConstruction:   
    def __init__(self: Self, limit: int = 10) -> None:
        self.limit = limit
        
    def __call__(self: Self, solution: Solution) -> Solution: 
        while (c := max(filter(lambda v: v[0] is not None,
                        map(lambda c: (solution.upper_bound_increment_add(c), c),
                            islice(solution.heuristic_add_moves(), self.limit))),
                    default = None,
                    key = itemgetter(0))) is not None:
            solution.add(c[1])
        return solution
                
    class SolutionProtocol(Protocol[T, Component]):
        def add(self: Self, component: Component) -> None: ...
        def heuristic_add_moves(self: Self) -> Iterable[Component]: ...  
        def upper_bound_increment_add(self: Self, component: Component) -> Optional[T]: ...
        
    Solution = TypeVar('Solution', bound=SolutionProtocol)
     
class HGRASP:
    def __init__(self: Self, alpha: Optional[float] = 0.1,
                 seed: Optional[int] = None,
                 local_search: Optional[LocalSearch] = None, 
                 **kwargs: Unpack[TypedDict]) -> None:
        self.alpha = alpha
        self.seed = seed
        self.local_search = local_search
        self.kwargs = kwargs
         
    def __call__(self: Self, solution: Solution, timer: Timer) -> Optional[Solution]:
        best, bobjv = None, None
        while not timer.finished():
            s = solution.copy()
            b, bobj = (s.copy(), s.objective()) if s.feasible() else (None, None)
            candidates = [(cast(T, s.heuristic_value(c)), c) for c in s.add_moves()]
            while len(candidates) != 0:
                cmin = min(candidates, key=itemgetter(0))[0]
                cmax = max(candidates, key=itemgetter(0))[0]
                thresh = cmin + self.alpha * (cmax - cmin)
                rcl = [c for decr, c in candidates if decr <= thresh]
                c = random.choice(rcl)
                s.add(c)
                if s.feasible():
                    obj = cast(T, s.objective())
                    if bobj is None or obj > bobj:
                        b, bobj = s.copy(), obj
                candidates = [(cast(T, s.heuristic_value(c)), c) for c in s.add_moves()]
            if b is not None:
                if self.local_search is not None:
                    b: HGRASP.Solution = self.local_search(b, **self.kwargs)
                    if b is not None and b.feasible():
                        bobj = cast(T, b.objective())
                if bobjv is None or bobj > bobjv:
                    best, bobjv = b, bobj     
                    logging.debug("SCORE: ", solution.score())
        return best
     
    class SolutionProtocol(Protocol[T, Component]):
        def copy(self: Self) -> Self: ...
        def feasible(self: Self) -> bool: ... 
        def objective(self: Self) -> Optional[T]: ...
        def add_moves(self: Self) -> Iterable[Component]: ...
        def heuristic_value(self: Self, component: Component) -> Optional[T]: ...
        def add(self: Self, component: Component) -> None: ...
    
    Solution = TypeVar('Solution', bound=SolutionProtocol)
    
    class LocalSearchProtocol(Protocol[Solution]):  
        def __call__(self: Self, 
                     solution: HGRASP.Solution,
                     **kwargs: Unpack[TypedDict]) -> Optional[HGRASP.Solution]: ... 
        
    LocalSearch = TypeVar('LocalSearch', bound=LocalSearchProtocol) 