from __future__ import annotations

import math
import time
import random

from typing import Optional, Iterable, Any, TypeVar
from typing_extensions import Self

from dataclasses import dataclass
from operator import itemgetter

T = TypeVar('T')

class Timer:
    def __init__(self: Self, budget: Optional[float] = None) -> None:
        self.budget = math.inf if budget is None else budget 
        self.start = time.perf_counter()
    
    def budget(self: Self) -> float:
        return self.budget

    def elapsed(self: Self) -> float:
        return time.perf_counter() - self.start

    def finished(self) -> bool:
        return self.elapsed() > self.budget

@dataclass 
class ConstantDecay:
    alpha: float
    def __call__(self, t: float) -> float:
        return t * self.alpha

@dataclass 
class LinearDecay:
    init_temperature: float
    def __call__(self, t: float) -> float:
        return t * self.init_temperature

@dataclass
class ExponentialAcceptance:
    def __call__(self: Self, delta: float, temperature: float) -> float:
        return math.exp(- delta / temperature) if delta > 0 else 1.0
     
def argmax(seq: Iterable[T]) -> int:
    return max(enumerate(seq), key=itemgetter(1))[0]

def isclose(a, b, rel_tol = 1e-6, abs_tol = 1e-9):
    return math.isclose(a, b, rel_tol = rel_tol, abs_tol = abs_tol)
    
   
def non_repeating_lcg(n: int, seed: Optional[Any] = None) -> Iterable[int]:
    if seed is not None:
        random.seed(seed)
    "Pseudorandom sampling without replacement in O(1) space"
    if n > 0:
        a = 5 # always 5
        m = 1 << math.ceil(math.log2(n))
        if m > 1:
            c = random.randrange(1, m, 2)
            x = random.randrange(m)
            for _ in range(m):
                if x < n: 
                    yield x
                x = (a * x + c) % m
        else:
            yield 0