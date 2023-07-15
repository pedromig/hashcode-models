import random

from ..problem import Problem
from ..solution import Solution
from .util import debug


def enum_first(problem: Problem) -> Solution:
    solution = problem.empty_solution() 
    while True:
        try:
            c = next(solution.enum_add_move())
            solution.add(c)
            debug(f"SCORE: {solution.score()}, UB: {solution.upper_bound()}")
        except StopIteration:
            break
    return solution

def enum_first_elite(problem: Problem) -> Solution:
    solution = problem.empty_solution()
    best, bobjv = (
        solution.copy(),
        solution.objective_value() if solution.feasible() else (None, None),
    )
    c = next(solution.enum_add_move())
    while True:
        try:
            solution.add(c)
            if solution.feasible():
                obj = solution.objective_value()
                if bobjv is None or obj > bobjv:
                    best, bobjv = solution.copy(), obj
                    debug(f"SCORE: {solution.score()}, UB: {solution.upper_bound()}")
            c = next(solution.enum_add_move())
        except StopIteration:
            break
    return best


def enum_heuristic_first(problem: Problem) -> Solution:
    solution = problem.empty_solution()
    while True:
        try:
            c = next(solution.enum_heuristic_add_move()) 
            solution.add(c) 
            debug(f"SCORE: {solution.score()}, UB: {solution.upper_bound()}")
        except StopIteration:
            break
    return solution


def enum_heuristic_first_elite(problem: Problem) -> Solution:
    solution = problem.empty_solution()
    best, bobjv = (
        (solution.copy(), solution.objective_value())
        if solution.feasible()
        else (None, None)
    )
    c = next(solution.enum_heuristic_add_move())
    while c is not None:
        try:
            solution.add(c)
            if solution.feasible():
                obj = solution.objective_value()
                if bobjv is None or obj > bobjv:
                    best, bobjv = solution.copy(), obj
                    debug(f"SCORE: {solution.score()}, UB: {solution.upper_bound()}")
            c = next(solution.enum_heuristic_add_move())
        except StopIteration:
            break
    return best


def enum_random(problem: Problem) -> Solution:
    solution = problem.empty_solution()
    while True:
        cs = list(solution.enum_add_move())
        if len(cs) > 0:
            solution.add(random.choice(cs))
            debug(f"SCORE: {solution.score()}, UB: {solution.upper_bound()}")
        else:
            break
    return solution


def enum_heuristic_random(problem: Problem) -> Solution:
    solution = problem.empty_solution()
    while True:
        cs = list(solution.enum_heuristic_add_move())
        if len(cs) > 0:
            solution.add(random.choice(cs))
            debug(f"SCORE: {solution.score()}, UB: {solution.upper_bound()}")
        else:
            break
    return solution


def enum_best_objv_increment(problem: Problem) -> Solution:
    solution = problem.empty_solution()
    while True:
        cs = list(solution.enum_add_move())
        if len(cs) > 0:
            bc, bincr = cs[0], solution.objective_increment_add(cs[0])
            for c in cs[1:]:
                incr = solution.objective_increment_add(c)
                if incr > bincr:
                    bc, bincr = c, incr
            solution.add(bc)
            debug(f"SCORE: {solution.score()}, UB: {solution.upper_bound()}")
        else:
            break
    return solution


def enum_best_heuristic_objv_increment(problem: Problem) -> Solution:
    solution = problem.empty_solution()
    while True:
        cs = list(solution.enum_heuristic_add_move())
        if len(cs) > 0:
            bc, bincr = cs[0], solution.objective_increment_add(cs[0])
            for c in cs[1:10]:
                incr = solution.objective_increment_add(c)
                if incr > bincr:
                    bc, bincr = c, incr
            solution.add(bc)
            debug(f"SCORE: {solution.score()}, UB: {solution.upper_bound()}")
        else:
            break
    return solution


def enum_best_ub_increment(problem: Problem) -> Solution:
    solution = problem.empty_solution()
    while True:
        components = list(solution.enum_add_move())
        if len(components) > 0:
            best_component = components[0]
            best_increment = solution.upper_bound_increment_add(best_component)
            for component in components[1:]:
                aux = solution.upper_bound_increment_add(component)
                if aux > best_increment:
                    best_component = component
                    best_increment = aux
                if aux == 0:
                    break
            # print(best_component)
            solution.add(best_component)
            debug(f"SCORE: {solution.score()}, UB: {solution.upper_bound()}")
        else:
            break
    return solution


def enum_best_heuristic_ub_increment(problem: Problem) -> Solution:
    solution = problem.empty_solution()
    while True:
        components = list(solution.enum_heuristic_add_move())
        if len(components) > 0:
            best_component = components[0]
            best_increment = solution.upper_bound_increment_add(best_component)
            for component in components[1:]:
                aux = solution.upper_bound_increment_add(component)
                if aux > best_increment:
                    best_component = component
                    best_increment = aux
            solution.add(best_component)
            debug(f"SCORE: {solution.score()}, UB: {solution.upper_bound()}")
        else:
            break
    return solution


def enum_best_heuristic_ub_increment_limit(
    problem: Problem, limit: int = 10
) -> Solution:
    solution = problem.empty_solution()
    while True:
        bc, bincr = None, None
        for l, c in enumerate(solution.enum_heuristic_add_move()):
            if limit is not None and l == limit:
                break
            incr = solution.upper_bound_increment_add(c)
            if bc is None or incr > bincr:
                bc, bincr = c, incr
        debug(f"SCORE: {solution.score()}, UB: {solution.upper_bound()}")
        if bc is None:
            break
        solution.add(bc)
    return solution


def greedy_construction(problem: Problem) -> Solution:
    solution = problem.empty_solution()
    best, bobjv = (
        (solution.copy(), solution.objective_value())
        if solution.feasible()
        else (None, None)
    )
    ci = solution.enum_add_move()
    c = next(ci, None)
    while c is not None:
        bc, bincr = c, solution.upper_bound_increment_add(c)
        for c in ci:
            incr = solution.upper_bound_increment_add(c)
            if incr > bincr:
                bc, bincr = c, incr
        solution.add(bc)
        if solution.feasible():
            obj = solution.objective_value()
            if bobjv is None or obj > bobjv:
                best, bobjv = solution.copy(), obj
                debug(f"SCORE: {solution.score()}, UB: {solution.upper_bound()}")
        ci = solution.enum_add_move()
        c = next(ci, None)
    return best
