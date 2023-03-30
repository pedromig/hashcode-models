import time

from model import Problem, Solution 

def first_improvement(solution: Solution, timer) -> Solution: 
    ZERO = tuple([0] * len(solution.objective_value()))
    while not timer.finished():
        for move in solution.enum_local_move():
            incr = solution.objective_increment_local(move)
            if incr > ZERO:
                solution.step(move)
                break
        print(f"INCR: {incr}, SCORE: {solution.score()}")
    return solution

def best_improvement(solution: Solution, timer):
    while not timer.finished():
        best = None
        bestincr = 0
        for move in solution.enum_local_move():
            incr = solution.objective_increment_local(move)
            if incr >= bestincr:
                best = move
                bestincr = incr
        if best is None:
            break
        else:
            solution.step(best)
