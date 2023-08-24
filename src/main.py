#!/usr/bin/env pypy3

# Dirty Trick to import nasf4nio
import sys
sys.path.append("..")

import nasf4nio.solvers as solvers

import datacenter.a as model

def debug(*args: object) -> None:
    print(*args, file=sys.stderr)
    
    
import nasf4nio as nio
 
if __name__ == "__main__": 
    # Problem    
    problem = model.Problem.from_stdin()
    
    # Solution 
    # solution = problem.empty_solution() 
    # solution = problem.random_solution()
        
    # Solvers 
     
    # Constructive
    
    # solver = solvers.SimpleConstruction()
    
    # solver = solvers.HeuristicConstruction()
    # solver = solvers.NarrowGuidedHeuristicConstruction()
    # solver = solvers.GRASP()
    
    # solver = solvers.GreedyConstruction() 
    # solver = solvers.GreedyUpperBoundConstruction()
    # solver = solvers.GreedyObjectiveConstruction()
    
    # solver = solvers.BeamSearch()
    # solver = solvers.GRASP()
    # solver = solvers.IteratedGreedy() 
    # solver = solvers.MMAS()
     
    # solution: model.Solution = solver(solution)
    
    # Local
    
    # solver = solvers.FirstImprovement()
    # solver = solvers.DeterministicFirstImprovement()
    # solver = solvers.BestImprovement()
    # solver = solvers.SimulatedAnnealing()
    # solver = solvers.TabuSearch()
    # solver = solvers.ILS()
    # solver = solvers.RLS()
    
    # solution: model.Solution = solver(solution)
    
    # Tests
     
    # print(solution) 
    # debug("SCORE:", solution.score()) 
    # debug("OBJECTIVE VALUE:", solution.objective_value()) 
    # debug("UPPER BOUND:", solution.upper_bound())  
 