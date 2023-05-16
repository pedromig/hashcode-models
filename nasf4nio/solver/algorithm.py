import operator
import random

from ..problem import Problem
from ..solution import Solution

from .util import Timer, debug


# Beam Search
def beam_search(problem: Problem, timer: Timer, beam_width: int = 10) -> Solution:
    solution = problem.empty_solution()
    best, bobjv = (
        (solution, solution.objective_value()
         ) if solution.feasible() else (None, None)
    )
    l0 = [(solution.upper_bound(), solution)]
    while not timer.finished():
        l1 = []
        for ub, s in l0:
            for c in s.enum_add_move():
                l1.append((ub + s.upper_bound_increment_add(c), s, c))
        if l1 == []:
            return best
        l1.sort(reverse=True, key=operator.itemgetter(0))
        l0 = []
        for ub, s, c in l1[:beam_width]:
            s: Solution = s.copy()
            s.add(c)
            if s.feasible():
                obj = s.objective_value()
                if bobjv is None or obj > bobjv:
                    best, bobjv = s, obj
                    debug(f"SCORE: {s.score()}, UB: {s.upper_bound()}")
            l0.append((ub, s))
    return best


# Iterated Greedy


# cmin = min(candidates, key=operator.itemgetter(0))[0]
# thresh = cmin + alpha * (cmax - cmin)
# print([i.server for i in rcl])
# print(c)
def iterated_greedy(
    problem: Problem, timer: Timer, alpha: float = 0.1, N: int = 10
) -> Solution:
    s = problem.empty_solution()
    best, bobjv = (s.copy(), s.objective_value()
                   ) if s.feasible() else (None, None)
    while not timer.finished():
        candidates = [(s.upper_bound_increment_add(c), c)
                      for c in s.enum_add_move()]
        while len(candidates) != 0:
            cmax = max(candidates, key=operator.itemgetter(0))[0]
            rcl = [c for decr, c in candidates if decr == cmax]
            c = random.choice(rcl)
            s.add(c)
            if bobjv is not None and s.objective_value() <= bobjv:
                break
            candidates = [
                (s.upper_bound_increment_add(c), c) for c in s.enum_add_move()
            ]

        if s.feasible():
            obj = s.objective_value()
            if bobjv is None or obj > bobjv:
                best, bobjv = s.copy(), obj
                debug(f"SCORE: {s.score()}, UB: {s.upper_bound()}")

        # Destruction
        for _ in range(N):
            c = s.random_remove_move()
            if c is None:
                break
            s.remove(c)
        debug(f"SCORE: {s.score()}, UB: {s.upper_bound()}")
    return best


# cmin = min(candidates, key=operator.itemgetter(0))[0]
# thresh = cmax - alpha * (cmax - cmin)
# print(s.upper_bound(), s.score())
# assert s.ub[0] == s.objv[0], f"{s.ub}\n{s.objv}"
def grasp(
    problem: Problem, timer: Timer,
    alpha=0.1, local_search=None, *args, **kwargs
):
    best, bobjv = None, None
    while not timer.finished():
        s = problem.empty_solution()
        b, bobj = (s.copy(), s.objective_value()
                   ) if s.feasible() else (None, None)
        candidates = [(s.upper_bound_increment_add(c), c)
                      for c in s.enum_add_move()]
        while len(candidates) != 0:
            cmax = max(candidates, key=operator.itemgetter(0))[0]
            rcl = [c for decr, c in candidates if decr == cmax]
            c = random.choice(rcl)
            s.add(c)
            if s.feasible():
                obj = s.objective_value()
                if bobj is None or obj > bobj:
                    b, bobj = s.copy(), obj
            candidates = [
                (s.upper_bound_increment_add(c), c) for c in s.enum_add_move()
            ]
        debug(f"SCORE: {s.score()}, UB: {s.upper_bound()}")
        if b is not None:
            if local_search is not None:
                local_search(b, timer, *args, **kwargs)
                debug(f"SCORE: {s.score()}")

            if bobjv is None or bobj > bobjv:
                best, bobjv = b, bobj
                debug(f"SCORE: {s.score()}, UB: {s.upper_bound()}")
    return best
