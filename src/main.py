#!/usr/bin/env pypy3

# Dirty Trick to import nasf4nio
import sys
import functools

sys.path.append("..")

from nasf4nio.solver import *

# import datacenter.model.a as model
# import datacenter.model.b as model
# import datacenter.model.c as model

# import scanning.model.a as model
# import scanning.model.b as model
# import scanning.model.c as model
import scanning.model.d as model

# import delivery.model as model

if __name__ == "__main__":
    problem = model.Problem.from_stdin()
    # debug(repr(problem))
    # debug(problem) 
        
    # Default Solution 
    # solution = problem.empty_solution() 
    solution = problem.heuristic_solution()
    # solution = problem.random_solution()
     
    # Solvers

    # solution.add(model.Component(problem, book=0, library=1)) 
    # solution.add(model.Component(problem, book=5, library=1))
    # solution.add(model.Component(problem, book=2, library=1))    
    # solution.add(model.Component(problem, book=None, library=0))  
    # solution.add(model.Component(problem, book=None, library=1))  
    # solution.add(model.Component(problem, book=None, library=2))  
    # solution.add(model.Component(problem, book=None, library=3))  
    # solution.add(model.Component(problem, book=None, library=4))  
    # solution.add(model.Component(problem, book=1, library=0)) 
    # solution.add(model.Component(problem, book=4, library=0))  
    # solution.add(model.Component(problem, book=3, library=1)) 
    # print(solution.order, solution.iorder, solution.used, solution.start, solution.quota)
    # solution._insert(4, 1)
    # solution._reverse(1, 0)
    # print(solution.order, solution.iorder, solution.used, solution.start, solution.quota)
    # solution._reverse(3, 1)
    # print(solution.order, solution.iorder, solution.used, solution.start, solution.quota)
    # solution._reverse(3, 1)
    # print(solution.order, solution.iorder, solution.used, solution.start, solution.quota)
    
    # print(repr(solution))
   
    # solution.remove(model.Component(problem, book=3, library=0))
    # print(list(solution.enum_add_move()))
    # print(list(solution.enum_heuristic_add_move()))
      
    # solution = enum_first(problem)
    # solution = enum_first_elite(problem)    

    # solution = enum_heuristic_first(problem)
    # solution = enum_heuristic_first_elite(problem)

    # solution = enum_random(problem)
    # solution = enum_heuristic_random(problem)
    
    # solution = enum_best_objv_increment(problem)
    # solution = enum_best_heuristic_objv_increment(problem)
    
    # solution = greedy_construction(problem)

    # solution = enum_best_ub_increment(problem)
    # solution = enum_best_heuristic_ub_increment(problem)
    # solution = enum_best_heuristic_ub_increment_limit(problem, limit=10)
    
    # solution = beam_search(problem, Timer(60))
    # solution = iterated_greedy(problem, Timer(1800))

    # ZERO = tuple([0]* problem.p) 
    # best_improvement = functools.partial(first_improvement, zero=ZERO)
    # first_improvement = functools.partial(best_improvement, zero=ZERO)
    # ils = functools.partial(ils, zero=ZERO)
    # rls = functools.partial(rls, zero=ZERO)
    # simulated_annealing = functools.partial(simulated_annealing, zero=ZERO)
     
    # print(repr(solution.quota)) 
    debug("SCORE BEFORE LS:", solution.score()) 
    # solution = first_improvement(solution, Timer(1800))
    # solution = best_improvement(solution, Timer(60))
    # # solution = ils(solution, Timer(1800), kick=3)
    solution = rls(solution, Timer(1800))
    # # solution = simulated_annealing(solution, Timer(60))

    # # solution = grasp(problem, Timer(1800), local_search=None)
    # # print(len(solution.used), len(problem.scores), sum([i <= problem.d for i in solution.start]))
    
    # # print(list(solution.enum_heuristic_add_move()))
    # print(solution.order, solution.start)
    # print(len(solution.libraries))
    
    # debug("SCORE:", solution.score()) 
    # debug("OBJECTIVE VALUE:", solution.objective_value()) 
    # # debug("UPPER BOUND:", solution.upper_bound())  
 
    # # debug("SCORE (FULL):", solution._score())
    # debug("OBJECTIVE VALUE (FULL):", solution._objective_value()) 
    
    # debug(repr(solution))
    # print(solution)