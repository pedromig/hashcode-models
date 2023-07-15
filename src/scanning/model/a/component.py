from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from . import *

class Component:
  def __init__(self: Component, problem: Problem, book: int, library: int) -> None:
    self.problem = problem
    self.book = book
    self.library = library 
    
  def __lt__(self: Component, other: Component) -> bool:
      return (self.book, self.library) < (other.book, other.library)

  def __eq__(self: Component, other: Component) -> bool:
      return self.book == other.book and self.library == other.library
     
  def __repr__(self: Component):
    return self.__str__()
    
  def __str__(self: Component):
    return f"({self.book}, {self.library})"