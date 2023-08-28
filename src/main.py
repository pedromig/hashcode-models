#!/usr/bin/env pypy3

# Dirty Trick to import nasf4nio
import sys
sys.path.append("..")

import logging

import nasf4nio.solvers as solvers
from nasf4nio.utils import Timer

import datacenter.a as model
# import datacenter.b as model
# import datacenter.c as model

# import scanning.a as model
# import scanning.b as model

def debug(*args: object) -> None:
    print(*args, file=sys.stderr)
    
 
if __name__ == "__main__": 
    
    # Problem    
    problem = model.Problem.from_stdin() 
    # print(problem.servers, problem.sservers, problem.pservers)
     
    # Solution 
    # solution = problem.empty_solution() 
    solution = problem.random_solution()
        
    # Solvers 
     
    # Constructive
    
    # solver = solvers.SimpleConstruction()
    
    # solver = solvers.HeuristicConstruction()
    # solver = solvers.NarrowGuidedHeuristicConstruction()
    # solver = solvers.HGRASP()
    
    # solver = solvers.GreedyConstruction() 
    #solver = solvers.GreedyUpperBoundConstruction()
    # solver = solvers.GreedyObjectiveConstruction()
    
    # solver = solvers.BeamSearch()
    # solver = solvers.GRASP(alpha=None)
    # solver = solvers.IteratedGreedy(alpha=None) 
    # solver = solvers.MMAS(1 - 1/solution.upper_bound())

    # population = [solution.copy() for _ in range(10)]
    
    # solution: model.Solution = solver(population, Timer(1200)) 
    # solution: model.Solution = solver(solution, Timer(60))
    # solution: model.Solution = solver(solution)
    debug("SCORE:", solution.score()) 
    debug("OBJECTIVE: ", solution.objective())
    debug("UPPER BOUND:", solution.upper_bound())  
    
    # print(solution.upper_bound(), solution.objective(), solution.ss)
    # Local 
    # tuple_zero = 0 
    # tuple_zero = tuple([0] * problem.p)
    # tuple_zero = tuple([tuple([0] * problem.p) for _ in range(len(problem.rows))])
     
    # solver = solvers.FirstImprovement(zero=tuple_zero)
    # solver = solvers.DeterministicFirstImprovement(zero=tuple_zero)
    # solver = solvers.BestImprovement(zero=tuple_zero)
    # solver = solvers.SimulatedAnnealing(20)
    # solver = solvers.TabuSearch(zero=tuple_zero)
    # solver = solvers.ILS(zero=tuple_zero)
    # solver = solvers.RLS(zero=tuple_zero)
     
    # solution: model.Solution = solver(solution, Timer(300))
    
    # print(solution) 
    # debug("SCORE:", solution.score()) 
 
