from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from . import *


class LocalMove: 
  def __init__(self: LocalMove, problem: Problem, 
               add: tuple[int] | None = None, 
               remove: tuple[int] | None = None,
               swap: tuple[int] | None = None) -> None:
    self.problem = problem
    self.add = add
    self.remove = remove 
    self.swap = swap
    
  def __lt__(self: LocalMove, other: LocalMove):
      sadd = self.add if self.add is not None else tuple(),
      sremove = self.remove if self.remove is not None else tuple()
      sswap = self.swap  if self.swap is not None else tuple()
      add = other.add if other.add is not None else tuple(),
      remove = other.remove if other.remove is not None else tuple()
      swap = other.swap if other.swap is not None else tuple()
      return (sadd, sremove, sswap) < (add, remove, swap)

  def __eq__(self: LocalMove, other: LocalMove):
      return self.add == other.add and self.remove == other.remove and self.swap == other.swap
    
  def __repr__(self: LocalMove) -> str:
    return self.__str__() 
    
  def __str__(self: LocalMove) -> str:
    return f"({self.add}, {self.remove}, {self.swap})"