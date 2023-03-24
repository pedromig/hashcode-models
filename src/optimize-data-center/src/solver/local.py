import time

def first_improvement(solution, timer):
    while not timer.finished():
        for move in solution.enum_local_move():
            incr = solution.objective_increment_local(move)
            if incr > 0:
                solution.step(move)
                # print(solution.get_objective_value())
                break
        else: 
            break

def best_improvement(solution, timer):
    while not timer.finished():
        best = None
        bestincr = 0
        for move in solution.enum_local_move():
            incr = solution.get_objective_increment_local(move)
            if incr >= bestincr:
                best = move
                bestincr = incr
        if best is None:
            break
        else:
            solution.step(best)
