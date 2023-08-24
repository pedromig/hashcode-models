from __future__ import annotations

from typing import Generic, TypeVar, Protocol, Optional
from typing_extensions import Self

Component = TypeVar("Component")

class SimpleConstruction: 
    def __call__(self: Self, solution: Solution) -> Solution:
        while (c := next(solution.add_moves(), None)) is not None:
            solution.add(c)
        return solution 
    
    class SolutionProtocol(Protocol[Component]):
        def add_moves(self: Self) -> Optional[Component]: ...
        def add(self: Self, component: Component) -> None: ...  
        
    Solution = TypeVar("Solution", bound=SolutionProtocol) 