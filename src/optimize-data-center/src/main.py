#!/usr/bin/env pypy3

import random
import sys
import time

from model import Problem, Component
from solver import *

class Timer:
    def __init__(self, time_limit):
        self.time_limit = time_limit
        self.start = time.perf_counter()

    def elapsed(self):
        return time.perf_counter() - self.start
     
    def finished(self):
        return self.elapsed() > self.time_limit 
       
def debug(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)

if __name__ == "__main__":
    problem = Problem.from_stdin()
    # debug(problem)
    # debug(repr(problem))

    # solution = problem.empty_solution()
    # solution = problem.random_solution()
    # debug(solution)
    # debug(repr(solution))
   
    # Example (bound.in) 
    # solution = problem.empty_solution()
    # 
    # debug(solution.upper_bound())    
    # debug(solution.upper_bound_increment_add(Component(problem, 0, 0, 1)))   
    # solution.add(Component(problem, 0, 0, 1))  
    # debug(solution.upper_bound()) 

    # debug(solution.upper_bound_increment_add(Component(problem, 1, 0, 1)))
    # solution.add(Component(problem, 1, 0, 1))
    # debug(solution.upper_bound())
     
    # Solvers 

    # solution = greedy_construction(problem)
     
    # solution = enum_first(problem)
    # solution = enum_first_elite(problem)
     
    # solution = enum_heuristic_first(problem)
    # solution = enum_heuristic_first_elite(problem) 

    # solution = enum_random(problem) 
    # solution = enum_heuristic_random(problem) 
    
    # solution = enum_best_heuristic_objv_increment(problem)

    # solution = enum_best_ub_increment(problem)
    # solution = enum_best_heuristic_ub_increment(problem)
    solution = enum_best_heuristic_ub_increment_limit(problem, limit = 10)
    
    # solution = bs(problem)
    # solution = ig(problem)
    # solution = grasp(problem)

    debug("SCORE: ", solution.score())  
    debug("OBJECTIVE VALUE: ", solution.objective_value())
    debug("UPPER BOUND: ", solution.upper_bound())
