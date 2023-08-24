from __future__ import annotations

import random

from typing import cast, Optional, Union, TypeVar, Protocol, Iterable, Callable
from typing_extensions import Self

from ..utils import LinearDecay, ExponentialAcceptance

Number = Union[float, int]

class TimerProtocol(Protocol):
    def finished(self: Self) -> bool: ...
    def elapsed(self: Self) -> float: ...
    def budget(self: Self) -> float: ...
    
Timer = TypeVar('Timer', bound=TimerProtocol)

T = TypeVar('T', bound=Number, covariant=True)
LocalMove = TypeVar('LocalMove')

class SolutionProtocol(Protocol[T, LocalMove]):
    def copy(self: Self) -> Self: ...
    def objective(self: Self) -> T: ...
    def step(self: Self, move: LocalMove) -> None: ...
    def random_local_moves_wor(self: Self) -> Iterable[LocalMove]: ...
    def objective_increment_local(self: Self, move: LocalMove) -> Optional[T]: ...

Solution = TypeVar('Solution', bound=SolutionProtocol)

TemperatureDecay = Optional[Callable[[float], float]]
AcceptanceCriteria = Optional[Callable[[float, float], float]] 

class SimulatedAnnealing:
    def __init__(self: Self, temperature: float,
                 seed: Optional[int] = None,
                 decay: Optional[TemperatureDecay] = None,
                 acceptance: Optional[AcceptanceCriteria] = None) -> None:
        self.temperature = temperature
        self.decay = LinearDecay(self.temperature) if decay is None else decay
        self.acceptance = ExponentialAcceptance() if acceptance is None else acceptance
        
        if seed is not None:
            random.seed(seed)
  
    def __call__(self: Self, solution: Solution, timer: Timer) -> Solution:
        best = solution.copy()
        bobjv = cast(T, best.objective())
        while not timer.finished():
            for move in solution.random_local_moves_wor():
                if (t := self.decay(1 - timer.elapsed() / timer.budget())) <= 0:
                    break
                delta = cast(T, solution.objective_increment_local(move))
                if self.acceptance(delta, t) >= random.random():
                    solution.step(move)
                    obj = cast(T, solution.objective())
                    if bobjv is None or obj > bobjv:
                        best = solution.copy()
                        bobjv = obj
                    break
        return best