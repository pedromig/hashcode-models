from __future__ import annotations

import logging

from typing import cast, Protocol, TypeVar, Optional, Iterable, Mapping, TypedDict
from typing_extensions import Self,Unpack

from operator import itemgetter

Component = TypeVar('Component')

class Comparable(Protocol): 
    def __lt__(self: Self, other: Self) -> bool: ...

T = TypeVar("T", bound=Comparable, covariant=True)
 
class GreedyConstruction:  
    def __call__(self: Self, solution: Solution) -> Solution: 
        while (c := max(filter(lambda v: v[0] is not None,
                        map(lambda c:((
                                solution.upper_bound_increment_add(c),
                                solution.objective_increment_add(c)
                                ), c),
                            solution.add_moves())),
                    default = None,
                    key = itemgetter(0))) is not None:
            solution.add(c[1]) 
            logging.info(f"SCORE: {solution.score()}")
        return solution
                
    class SolutionProtocol(Protocol[T, Component]):
        def add(self: Self, component: Component) -> None: ...
        def add_moves(self: Self) -> Iterable[Component]: ...  
        def objective_increment_add(self: Self, component: Component) -> Optional[T]: ... 
        def upper_bound_increment_add(self: Self, component: Component) -> Optional[T]: ...
        
    Solution = TypeVar('Solution', bound=SolutionProtocol)

class GreedyUpperBoundConstruction: 
    def __call__(self: Self, solution: Solution) -> Solution: 
        while (c := max(filter(lambda v: v[0] is not None,
                        map(lambda c: (solution.upper_bound_increment_add(c), c),
                            solution.add_moves())),
                    default = None,
                    key = itemgetter(0))) is not None:
            solution.add(c[1]) 
            logging.debug(f"SCORE: {solution.score()}")
        return solution
                    
    class SolutionProtocol(Protocol[T, Component]):
        def add(self: Self, component: Component) -> None: ...
        def add_moves(self: Self) -> Iterable[Component]: ...  
        def upper_bound_increment_add(self: Self, component: Component) -> Optional[T]: ...
        
    Solution = TypeVar('Solution', bound=SolutionProtocol)
         
class GreedyObjectiveConstruction:  
    def __call__(self: Self, solution: Solution) -> Solution:
        while (c := max(filter(lambda v: v[0] is not None,
                        map(lambda c: (solution.objective_increment_add(c), c),
                            solution.add_moves())),
                    default = None,
                    key = itemgetter(0))) is not None:
            solution.add(c[1])
            logging.debug(f"SCORE: {solution.score()}")
        return solution
                
    class SolutionProtocol(Protocol[T, Component]):
        def add(self: Self, component: Component) -> None: ...
        def add_moves(self: Self) -> Iterable[Component]: ...  
        def objective_increment_add(self: Self, component: Component) -> Optional[T]: ...
        
    Solution = TypeVar('Solution', bound=SolutionProtocol) 