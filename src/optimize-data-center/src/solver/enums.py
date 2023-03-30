import random

from model import Problem, Solution 

def enum_first(problem: Problem) -> Solution:
    solution = problem.empty_solution()
    while True:
        try:
            c = next(solution.enum_add_move()) 
            solution.add(c)
        except StopIteration:
            break
    return solution


def enum_first_elite(problem: Problem) -> Solution:
    s = problem.empty_solution()
    best, bobjv = (s.copy(), s.objective_value() if s.feasible() else (None, None))
    c = next(s.enum_add_move())
    while True:
        try:
            s.add(c)
            if s.feasible():
                obj = s.objective_value()
                if bobjv is None or obj > bobjv:
                    best, bobjv = s.copy(), obj 
            c = next(s.enum_add_move())
        except StopIteration:
            break
    return best

def enum_heuristic_first(problem:  Problem) -> Solution:
    solution = problem.empty_solution()
    while True:
        try:
            c = next(solution.enum_heuristic_add_move())
            solution.add(c)
        except StopIteration:
            break
    return solution

def enum_heuristic_first_elite(problem: Problem) -> Solution: 
    s = problem.empty_solution()
    best, bobjv = (s.copy(), s.objective_value()) if s.feasible() else (None, None)
    c = next(s.enum_heuristic_add_move())
    while c is not None:         
        try:
            s.add(c)
            if s.feasible():
                obj = s.objective_value()
                if bobjv is None or obj > bobjv:
                    best, bobjv = s.copy(), obj
            c = next(s.enum_heuristic_add_move())
        except StopIteration:
            break
    return best
  
def enum_random(problem: Problem) -> Solution:
    solution = problem.empty_solution()
    while True:
        cs = list(solution.enum_add_move()) 
        if len(cs) > 0:
            solution.add(random.choice(cs))
        else:
            break
    return solution

def enum_heuristic_random(problem: Problem) -> Solution:
    solution = problem.empty_solution()
    while True:
        cs = list(solution.enum_heuristic_add_move()) 
        if len(cs) > 0:
            solution.add(random.choice(cs))
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
            print(f"SCORE: {solution.score()}, UB: {solution.upper_bound()}")
            solution.add(best_component)
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
            print(f"SCORE: {solution.score()}, UB: {solution.upper_bound()}")
        else: 
            break 
    return solution

def enum_best_heuristic_ub_increment_limit(problem: Problem, limit: int = 10) -> Solution:
    solution = problem.empty_solution()
    while True: 
        bc, bincr = None, None 
        for l, c in enumerate(solution.enum_heuristic_add_move()):
            if limit is not None and l == limit:
                break
            incr = solution.upper_bound_increment_add(c)
            if bc is None or incr > bincr:
                bc, bincr = c, incr
        if bc is None:
            break
        solution.add(bc)
    return solution


def greedy_construction(problem: Problem):
    s = problem.empty_solution()
    best, bestobj = (s.copy(), s.objective_value()) if s.feasible() else (None, None)
    ci = s.enum_add_move()
    c = next(ci, None)
    while c is not None:
        bestc, bestincr = c, s.upper_bound_increment_add(c)
        for c in ci:
            incr = s.upper_bound_increment_add(c)
            if incr > bestincr:
                bestc, bestincr = c, incr
        s.add(bestc)
        if s.feasible():
            obj = s.objective_value()
            if bestobj is None or obj > bestobj:
                best, bestobj = s.copy(), obj
        ci = s.enum_add_move()
        c = next(ci, None)
    return best
