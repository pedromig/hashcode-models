#!/usr/bin/env pypy3

# Dirty Trick to import nasf4nio
import sys
import functools

sys.path.append("..")

from nasf4nio.solver import *

import datacenter.model.a as model
# import datacenter.model.b as model
# import datacenter.model.c as model


if __name__ == "__main__":
    problem = model.Problem.from_stdin()
    # solution = problem.empty_solution()
    # solution = problem.random_solution() 
    solution = problem.empty_solution()

    # Solvers

    # solution = greedy_construction(problem)

    # solution = enum_first(problem)
    # solution = enum_first_elite(problem)

    # solution = enum_heuristic_first(problem)
    # solution = enum_heuristic_first_elite(problem)
    # print(solution.used)
    # print(solution.unused)

    # solution = enum_random(problem)
    # solution = enum_heuristic_random(problem)
    # solution = enum_best_heuristic_objv_increment(problem)

    # solution = enum_best_ub_increment(problem)
    # solution = enum_best_heuristic_ub_increment(problem)
    solution = enum_best_heuristic_ub_increment_limit(problem, limit=10)

    # solution = beam_search(problem, Timer(60))
    # solution = iterated_greedy(problem, Timer(60))

    ZERO = tuple([0]* problem.p) 
    # best_improvement = functools.partial(first_improvement, zero=ZERO)
    # first_improvement = functools.partial(best_improvement, zero=ZERO)
    ils = functools.partial(ils, zero=ZERO)
    # rls = functools.partial(rls, zero=ZERO)
    # simulated_annealing = functools.partial(simulated_annealing, zero=ZERO)

    # solution = first_improvement(solution, Timer(60))
    # solution = best_improvement(solution, Timer(1800))
    solution = ils(solution, Timer(1800), kick=3)
    # solution = rls(solution, Timer(1800))
    # solution = simulated_annealing(solution, Timer(60))

    # solution = grasp(problem, Timer(1800), local_search=None)

    # debug("SCORE:", solution.score())
    # debug("OBJECTIVE VALUE: ", solution.objective_value())
    # debug("UPPER BOUND: ", solution.upper_bound())
     
    # print(solution)
