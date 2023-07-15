
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from . import *
 
class LocalMove:
  def __init__(self: LocalMove, problem: Problem) -> None:
    ...