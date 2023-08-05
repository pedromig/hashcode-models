import math
import random

from typing import Any

from .. import Solution
from .util import Timer, debug


def first_improvement(solution: Solution, timer: Timer, zero: Any = 0) -> Solution:
    while not timer.finished():
        for move in solution.enum_local_move():
            incr = solution.objective_increment_local(move) 
            if incr > zero: 
                solution.step(move)
                debug(f"SCORE: {solution.score()}")
                break
    return solution

# print(bestincr, solution.objective_value())
# print(solution.objective_value())
# assert bestincr + solution.objective_value() == solution.objective_value() (after)


def best_improvement(solution: Solution, timer: Timer, zero: Any = 0):
    while not timer.finished():
        best = None
        bincr = zero
        for move in solution.enum_local_move():
            incr = solution.objective_increment_local(move)
            if incr >= bincr:
                best = move
                bincr = incr
        if best is None:
            break
        else:
            solution.step(best)
            debug(f"SCORE: {solution.score()}")
    return solution

# Random Local Search
def rls(solution: Solution, timer: Timer, zero: Any = 0) -> Solution:
    while not timer.finished():
        for move in solution.enum_random_local_move_wor():
            incr = solution.objective_increment_local(move)
            if incr >= zero:
                solution.step(move)
                debug(f"SCORE: {solution.score()}")
                break
        else:
            break
    return solution

# print(best.upper_bound())
# print(solution.upper_bound())
# print(f"SCORE (before kick): {solution.score()}")
# print(solution.upper_bound())
# print(f"SCORE (after kick): {solution.score()}")

# Iterated Local Search
def ils(solution: Solution, timer: Timer, kick: int = 3, zero: Any = 0) -> Solution:
    best = solution.copy()
    best_obj = best.objective_value()
    while not timer.finished():
        for move in solution.enum_local_move():
            incr = solution.objective_increment_local(move)
            if incr > zero:
                solution.step(move)
                break
            if timer.finished():
                if solution.objective_value() > best_obj:
                    return solution
                else:
                    return best
        else:
            obj = solution.objective_value()
            if obj >= best_obj:
                best = solution.copy()
                best_obj = obj
                debug(f"SCORE: {best.score()}")
            else:
                solution = best.copy()
            solution.perturb(kick)
    if solution.objective_value() > best_obj:
        return solution
    else:
        return best


# Simulated Annealing
def simulated_annealing(solution: Solution, timer: Timer, init_temp=None, acceptance=random.random(), zero: Any = 0):
    if init_temp is None:
        init_temp = timer.budget()

    def temperature(t): return t * init_temp
    def probability(incr, t): return 1 if incr > zero else math.exp(incr / t)

    best = solution.copy()
    bobjv = best.objective_value()

    while not timer.finished():
        for move in solution.enum_random_local_move_wor():
            t = temperature(1 - timer.elapsed() / timer.budget())
            if t <= 0:
                break
            incr = solution.objective_increment_local(move)
            if probability(incr, t) >= acceptance:
                solution.step(move)
                if solution.objective_value() > bobjv:
                    best, bobjv = solution.copy(), best.objective_value()
                    debug(f"SCORE: {best.score()}")
                break
        else:
            break
    return best
