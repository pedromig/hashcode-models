from __future__ import annotations

import random

from typing import Optional, Protocol, TypeVar 
from typing_extensions import Self

from dataclasses import dataclass

T = TypeVar("T")
Component = TypeVar("Component")

class SolutionProtocol(Protocol[T, Component]):
    def add(self: Self, component: Component) -> None: ...
    def objective(self: Self) -> T: ...
    def objective_increment_add(self: Self, component: Component) -> Optional[T]: ... 
    def random_add_move(self: Self) -> Optional[Component]: ...
    
Solution = TypeVar('Solution', bound=SolutionProtocol)

class ProblemProtocol(Protocol):
    def empty_solution(self: Self) -> Solution: ...
    
Problem = TypeVar('Problem', bound=ProblemProtocol)

def objective_increment_add_test(self: Problem, seed: Optional[int] = None) -> None:
    if seed is not None:
        random.seed(seed) 
        
    x: Solution = self.empty_solution()
    y: Solution = self.empty_solution()
    
    while (c := x.random_add_move()) is not None: 
        before, increment = x.objective(), x.objective_increment_add(c)  
        x.add(c)
        y.add(c)
        after = y.objective()
        assert (before + increment) == after, f"{before} (before) + {increment} (increment) != {after} (after) "
        