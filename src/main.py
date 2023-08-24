#!/usr/bin/env pypy3

# Dirty Trick to import nasf4nio
import sys
sys.path.append("..")

import nasf4nio.solvers as solvers
from nasf4nio.utils import Timer

import datacenter.a as model
# import datacenter.b as model
# import datacenter.c as model

def debug(*args: object) -> None:
    print(*args, file=sys.stderr)
    
 
if __name__ == "__main__": 
    # Problem    
    problem = model.Problem.from_stdin() 
     
    # Solution 
    solution = problem.empty_solution() 
    # solution = problem.random_solution()
        
    # Solvers 
     
    # Constructive
    
    # solver = solvers.SimpleConstruction()
    
    # solver = solvers.HeuristicConstruction()
    solver = solvers.NarrowGuidedHeuristicConstruction()
    # solver = solvers.HGRASP()
    
    # solver = solvers.GreedyConstruction() 
    # solver = solvers.GreedyUpperBoundConstruction()
    # solver = solvers.GreedyObjectiveConstruction()
    
    # solver = solvers.BeamSearch()
    # solver = solvers.GRASP(alpha=None)
    # solver = solvers.IteratedGreedy() 
    # solver = solvers.MMAS()
     
    solution: model.Solution = solver(solution) 
    debug("SCORE:", solution.score()) 
    debug("UPPER BOUND:", solution.upper_bound())  
    
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
    debug("SCORE:", solution.score()) 
 