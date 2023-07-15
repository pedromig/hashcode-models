from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from . import *


class LocalMove: 
  def __init__(self: LocalMove, problem: Problem, 
               i: tuple[int] | None = None, 
               j: tuple[int] | None = None,
               reverse: bool | None = None,
               swap: bool | None = None) -> None:
    self.problem = problem
    self.i, self.j = i, j
    self.reverse = reverse 
    self.swap = swap

  def __repr__(self: LocalMove) -> str:
    return self.__str__() 
    
  def __str__(self: LocalMove) -> str:
    return f"({self.i}, {self.j}, {self.swap}, {self.reverse})"