from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from . import *

class Component:
  def __init__(self: Component, problem: Problem, library: int) -> None:
    self.problem = problem
    self.library = library 
    
  def __lt__(self: Component, other: Component) -> bool:
      return self.library < other.library

  def __eq__(self: Component, other: Component) -> bool:
      return self.library == other.library
     
  def __repr__(self: Component):
    return self.__str__()
    
  def __str__(self: Component):
    return f"Component({self.library})"