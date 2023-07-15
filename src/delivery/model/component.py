from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from . import *

class Component:
  def __init__(self: Component, problem: Problem, library: int, book: int) -> None:
    self.problem = problem
    self.library = library
    self.book = book
    
  def __repr__(self: Component) -> str:
    return self.__str__()
  
  def __str__(self: Component) -> str:
    return f"({self.library}, {self.book})"