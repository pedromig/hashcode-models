from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from typing import Optional, Hashable, Iterable, TypeVar, Any, Generic
from typing_extensions import Self

T = TypeVar("T")

class Problem(ABC):
    def empty_solution(self: Self) -> Solution:
        raise NotImplementedError
  
    def random_solution(self: Self, seed: Optional[Any] = None) -> Solution:
        raise NotImplementedError    
    
P = TypeVar("P", bound=Problem)

class Component(ABC): 
    def id(self: Self) -> Hashable: ...
    
C = TypeVar("C", bound=Component)

class LocalMove(ABC):
    ...
    
LM = TypeVar("LM", bound=LocalMove)

class Solution(ABC, Generic[T, P, C, LM]):  
    def copy(self: Self) -> Self:
        raise NotImplementedError
   
    def feasible(self: Self) -> bool:
        raise NotImplementedError
    
    def objective(self: Self) -> Optional[T]:
        raise NotImplementedError
    
    def upper_bound(self: Self) -> Optional[T]:
        raise NotImplementedError
    
    def components(self: Self) -> Iterable[C]:
        raise NotImplementedError
    
    def add_moves(self: Self) -> Iterable[C]:
        raise NotImplementedError

    def heuristic_add_moves(self: Self) -> Iterable[C]:
        raise NotImplementedError

    def remove_moves(self: Self) -> Iterable[C]:
        raise NotImplementedError

    def local_moves(self: Self) -> Iterable[LM]:
        raise NotImplementedError

    def random_local_moves_wor(self: Self) -> Iterable[LM]:
        raise NotImplementedError

    def random_add_move(self: Self) -> Optional[C]:
        raise NotImplementedError

    def random_remove_move(self: Self) -> Optional[C]:
        raise NotImplementedError

    def random_local_move(self: Self) -> Optional[LM]:
        raise NotImplementedError

    def add(self: Self, component: C) -> None:
        raise NotImplementedError

    def remove(self: Solution, component: C) -> None:
        raise NotImplementedError
     
    def step(self: Self, move: LM) -> None:
        raise NotImplementedError 

    def perturb(self: Self) -> None:
        raise NotImplementedError

    def heuristic_value(self: Self, component: C) -> Optional[T]:
        raise NotImplementedError
    
    def objective_increment_local(self: Self, move: LM) -> Optional[T]:
        raise NotImplementedError

    def objective_increment_add(self: Self, component: C) -> Optional[T]:
        raise NotImplementedError

    def objective_increment_remove(self: Self, component: C) -> Optional[T]:
        raise NotImplementedError

    def upper_bound_increment_add(self: Self, component: C) -> Optional[T]:
        raise NotImplementedError

    def upper_bound_increment_remove(self: Self, component: C) -> Optional[T]:
        raise NotImplementedError