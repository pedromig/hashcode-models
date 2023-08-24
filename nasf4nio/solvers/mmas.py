from __future__ import annotations

import math
import random

from typing import cast, TypeVar, Protocol, Optional, Union, Iterable, Hashable, TypedDict, List
from typing_extensions import Self, Unpack

from collections import defaultdict

from ..utils import argmax, isclose

class TimerProtocol(Protocol):
    def finished(self: Self) -> bool: ...
    
Timer = TypeVar('Timer', bound=TimerProtocol)

T = TypeVar('T', bound=Union[float, int], covariant=True)

class ComponentProtocol(Protocol):
    def id(self: Self) -> Hashable: ...

Component = TypeVar('Component', bound=ComponentProtocol)

class SolutionProtocol(Protocol[T, Component]):
    def upper_bound(self: Self) -> Optional[T]: ...
    def objective(self: Self) -> Optional[T]: ...
    def copy(self: Self) -> Self: ...
    def feasible(self: Self) -> bool: ...
    def add_moves(self: Self) -> Iterable[Component]: ...
    def upper_bound_increment_add(self, component: Component) -> Optional[T]: ...
    def add(self: Self, component: Component) -> None: ...
    def components(self: Self) -> Iterable[Component]: ...

Solution = TypeVar('Solution', bound=SolutionProtocol)
     
class LocalSearchProtocol(Protocol[Solution]):  
    def __call__(self: Self, solution: Solution, **kwargs: Unpack[TypedDict]) -> Optional[Solution]: ... 
    
LocalSearch = TypeVar('LocalSearch', bound=SolutionProtocol)

Population = List[Solution]
     
class MMAS:
    def __init__(self: Self, tau_max: float, a: float = 5.0,
                 alpha: float = 1.0, beta: float = 3.0, rho: float = 0.5,
                 global_ratio: float = 0.5, n_restart: int = 500,
                 seed: Optional[int] = None,
                 local_search: Optional[LocalSearch[Solution]] = None,
                 **kwargs: Unpack[TypedDict]) -> None:
        self.tau_max = tau_max
        self.a = a
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.global_ratio = global_ratio
        self.n_restart = n_restart
        self.local_search = local_search
        self.kwargs = kwargs
        
        if seed is not None:
            random.seed(seed)

    def ant(self: Self, solution: Solution, tau: dict[T, float]) -> Solution:
        while True:
            cs, cszero, p = [], [], []
            best = None
            for c in solution.add_moves():
                k = cast(Component, c).id()
                incr = solution.upper_bound_increment_add(c)
                if incr is None:
                    raise ValueError("Upper Bound Increment cannot be NoneType")
                if isclose(incr, 0.0):
                    cszero.append(c)
                else:
                    cs.append(c)
                    p.append((tau[k]**self.alpha) * ((1.0 / incr) ** self.beta))
            if best is None:
                if len(cszero) > 0:
                    best = random.choice(cszero)
                elif len(cs) > 0:
                    if sum(p) > 0:
                        best = random.choices(cs, p, k = 1)[0]
                    else:
                        best = random.choice(cs)
                else:
                    break
            solution.add(best)
        return solution

    def __call__(self: Self, population: Population, timer: Timer) -> Optional[Solution]:
        tau_min = self.tau_max / self.a
        tau0 = self.tau_max
        tau: dict[Hashable, float] = defaultdict(lambda: tau0)

        best, bobjv = None, None
        ni = 0
        while not timer.finished():
            ni += 1
            
            # Build ants
            ants = []
            for s in population:
                if timer.finished():
                    break
                ant: Solution = self.ant(cast(Solution, s).copy(), tau)

                if ant.feasible() and (bobjv is None or ant.objective() > bobjv):
                    best = ant.copy()
                    bobjv = cast(T, ant.objective())
                    ni = 0

                if self.local_search is not None:
                    ant = self.local_search(ant, **self.kwargs)
                    if ant.feasible() and (bobjv is None or cast(T, ant.objective()) > bobjv):
                        best = ant.copy()
                        bobjv = cast(T, ant.objective())
                        ni = 0

                if ant.feasible():
                    ants.append(ant)

            # Update pheromones
            if ni < self.n_restart:
                tau_max = 1.0 / cast(T, bobjv)
                tau_min = tau_max / self.a
                tau0 = (1.0 - self.rho) * tau0
                tau0 = max(tau_min, min(tau_max, tau0))
                
                for k in tau:
                    tau[k] = max(tau_min, min(tau_max, (1.0 - self.rho) * tau[k]))
                    
                if (best is None and len(ants) > 0) or (len(ants) > 0 and random.random() > self.global_ratio):
                    # Using iteration-best ant
                    bi = argmax(map(lambda ant: cast(T, cast(Solution, ant).objective()), ants))
                    b: Solution = ants[bi]
                    obj = cast(T, b.objective())
                    for c in b.components():
                        cid = cast(Component, c).id()
                        if cid not in tau:
                            tau[cid] = min(tau_max, tau0 + 1.0 / obj)
                        else:
                            tau[cid] = min(tau_max, tau[cid] + 1.0 / obj)
                elif best is not None:
                    # Using global-best ant
                    for c in best.components():
                        cid = cast(Component, c).id()
                        if cid not in tau:
                            tau[cid] = min(tau_max, tau0 + 1.0 / cast(T, bobjv))
                        else:
                            tau[cid] = min(tau_max, tau[cid] + 1.0 / cast(T, bobjv))
            else:
                # Reinitialization
                tau_max = 1.0 / cast(T, bobjv)
                tau_min = tau_max / self.a
                tau0 = tau_max
                for k in tau:
                    tau[k] = tau0
        return best