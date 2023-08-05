import operator
import random

from ..problem import Problem
from ..solution import Solution

from .util import Timer, debug

# Heuristic Grasp
def hgrasp(problem: Problem, timer: Timer, alpha=0.1, local_search=None, *args, **kwargs):
    best, bobjv = None, None
    while not timer.finished():
        s = problem.empty_solution()
        b, bobj = (s.copy(), s.objective_value()) if s.feasible() else (None, None)
        candidates = [(s.heuristic_value(c), c) for c in s.enum_add_move()]
        while len(candidates) != 0:
            cmin = min(candidates, key=operator.itemgetter(0))[0]
            cmax = max(candidates, key=operator.itemgetter(0))[0] 
            thresh = cmax - alpha * (cmax - cmin)
            rcl = [c for decr, c in candidates if decr <= thresh]
            c = random.choice(rcl)
            s.add(c)
            if s.feasible():
                obj = s.objective_value() 
                if bobj is None or obj > bobj:
                    b, bobj = s.copy(), obj
            candidates = [(s.heuristic_value(c), c) for c in s.enum_add_move()]
        if b is not None:
            if local_search is not None:
                local_search(b, timer, *args, **kwargs)
                debug(f"SCORE: {s.score()}")
            
            if bobjv is None or bobj > bobjv:
                best, bobjv = b, bobj
                debug(f"SCORE: {s.score()}")
    return best