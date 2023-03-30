from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .problem import *
    
class LocalMove:
    def __init__(self: LocalMove, problem: Problem, 
                 add: tuple[int] | None, remove: tuple[int] | None, 
                 inplace: bool = False, pool: bool = False) -> None:
        self.problem = problem
        self.add = add
        self.remove = remove
        self.inplace = inplace
        self.pool = pool

    def __str__(self: LocalMove) -> str:
        return f"{self.add} {self.remove} {self.inplace} {self.pool}" 

    def __repr__(self: LocalMove) -> str:
        return self.__str__()
