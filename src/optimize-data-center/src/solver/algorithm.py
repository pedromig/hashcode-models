import time
import operator
import random

from model import Problem, Solution 

# SA

# Beam Search
def bs(problem, bw=10):
    s = problem.empty_solution()
    best, bestobj = (s, s.objective_value()) if s.feasible() else (None, None)
    L0 = [(s.upper_bound(), s)]
    while True:
        L1 = []
        for ub, s in L0:
            for c in s.enum_add_move():
                L1.append((ub + s.upper_bound_increment_add(c), s, c))
        if L1 == []:
            return best
        L1.sort(reverse=True, key=operator.itemgetter(0))
        L0 = []
        for ub, s, c in L1[:bw]:
            s = s.copy()
            s.add(c)
            if s.feasible():
                obj = s.objective_value()
                if bestobj is None or obj > bestobj:
                    best, bestobj = s, obj
            L0.append((ub, s))
    return best

# Iterated Greedy
def ig(problem, budget=15, alpha = 0.1, N = 10):
    start = time.perf_counter()
    s = problem.empty_solution()
    best, bestobj = (s.copy(), s.objective_value()) if s.feasible() else (None, None)
    while time.perf_counter() - start < budget:
        candidates = [(s.upper_bound_increment_add(c), c) for c in s.enum_add_move()]
        while len(candidates) != 0:
            cmin = min(candidates, key=operator.itemgetter(0))[0]
            cmax = max(candidates, key=operator.itemgetter(0))[0]
            thresh = cmin + alpha * (cmax - cmin)
            rcl = [c for decr, c in candidates if decr <= thresh]
            c = random.choice(rcl)
            s.add(c)
            if bestobj is not None and s.objective_value() <= bestobj:
                break
            candidates = [(s.upper_bound_increment_add(c), c) for c in s.enum_add_move()]
        if s.feasible():
            obj = s.objective_value() 
            if bestobj is None or obj > bestobj:
                best, bestobj = s.copy(), obj
            print(obj, bestobj)
        s = best.copy()

        # Destruction
        for _ in range(N): 
            c = s.random_remove_move()
            if c is None:
                break
            s.remove(c)
    return best
   
# Greedy Randomized Adaptive Search Procedure
def grasp(problem, budget=15, alpha=0.1, local_search = None, *args, **kwargs):
    start = time.perf_counter()
    best, bestobj = None, None
    while time.perf_counter() - start < budget:
        s = problem.empty_solution()
        b, bobj = (s.copy(), s.objective_value()) if s.feasible() else (None, None)
        candidates = [(s.upper_bound_increment_add(c), c) for c in s.enum_add_move()]
        while len(candidates) != 0:
            cmin = min(candidates, key=operator.itemgetter(0))[0]
            cmax = max(candidates, key=operator.itemgetter(0))[0]
            thresh = cmin + alpha * (cmax - cmin)
            c = random.choice([c for decr, c in candidates if decr <= thresh])
            s.add(c)
            if s.feasible():
                obj = s.objective_value()
                if bobj is None or obj > bobj:
                    b, bobj = s.copy(), obj
            candidates = [(s.upper_bound_increment_add(c), c) for c in s.enum_add_move()]
        if b is not None:
            if local_search is not None:
                local_search(b, *args, **kwargs)
            bobj = b.objective_value()
            if bestobj is None or bobj > bestobj:
                best, bestobj = b, bobj
    return best

# Random Local Search
def rls(solution, timer):
    while not timer.finished():
        for move in solution.enum_random_local_move_without_replacement():
            incr = solution.get_objective_increment_local(move)
            if incr >= 0:
                solution.step(move)
                break
        else: 
            break

# Iterated Local Search 
def ils(solution, budget):
    start = time.perf_c.inter()
    kickStrength = 3    # Suitable values may be 3 to 5
    best = solution.copy()
    best_obj = best.get_objective_value()
    while time.perf_c.inter() - start < budget:
        for move in solution.enum_local_move():
        # for move in solution.randomLocalMoveWOR():
            incr = solution.get_objective_increment_local(move)
            if incr > 0:
                solution.step(move)
                break
            if time.perf_c.inter() - start >= budget:
                if solution.get_objective_value() > best_obj:
                    return solution
                else:
                    return best
        else:
            # Local optimum found
            obj = solution.get_objective_value()
            if obj >= best_obj:
                best = solution.copy()
                best_obj = obj
                # print(best_obj, file=sys.stderr)
            else:
                solution = best.copy()
            for _ in range(kickStrength):
                solution.step(solution.random_local_move())
    if solution.get_objective_value() > best_obj:
        return solution
    else:
        return best
 
# EA parameters
# Nind     -> Population size
# Ngen     -> Number of generations
# nruns    -> Number of runs
# SP       -> Selective pressure, a number between 1 and 2
# Suggestion: try out different mutation rates and see what happens!
def ea(problem, budget, Nind = 400, Ngen = 10000, nruns = 1, SP = 2):
    start = time.perf_c.inter()
    # Mutation-only configuration
    # Slightly below the theoretical error threshold
    Px = 0
    Pm = (1 - 1./SP)*0.8
    # Slightly above the theoretical error threshold
    # Px = 0
    # Pm = (1 - 1./SP)*1.1

    # Mutation and crossover
    # Pm = (1 - 1./SP)*0.7; Px = 0.6

    def argsort(seq):
        return map(operator.itemgetter(1), sorted(zip(seq,range(len(seq)))))

    def cumsum(seq):
        s, cs = 0, []
        for x in seq:
            s += x
            cs.append(s)
        return cs

    # Mutation
    def mutate(pop, Pm):
        for s in pop:
            if random.random() < Pm:
                s.step(s.random_local_move())

    # Linear ranking (assuming maximization)
    def ranking(obj, sp=2):
        Nind = len(obj)
        fitness = Nind * [0]
        ix = list(argsort(obj))
        for i in range(Nind):
            fitness[ix[i]] = (sp-1.) * 2. * i / (Nind-1.)
        return fitness

    # SUS
    def sus(fitness, Nsel=None):
        Nind = len(fitness)
        if Nsel is None:
            Nsel = Nind
        cumfit = cumsum(fitness)
        cumfit = [x * Nsel / cumfit[-1] for x in cumfit]
        ix = []
        ptr = random.random()
        i = 0
        while i < Nind:
            if ptr < cumfit[i]:
                ix.append(i)
                ptr += 1
            else:
                i += 1
        random.shuffle(ix)
        return ix

    best = []
    best_solution = None
    best_solution_value = None
    for r in range(nruns):
        # Initialise random population
        pop = [problem.random_solution() for _ in range(Nind)]
        best.append([])
        i = 0
        while i < Ngen and time.perf_c.inter() - start < budget:
            # Evaluate individuals
            obj = [s.get_objective_value() for s in pop]
            best[r].append(max(obj))

            for s in pop:
                aux = s.get_objective_value()
                if best_solution_value is None or aux > best_solution_value:
                    best_solution = s.copy()
                    best_solution_value = s.get_objective_value()

            # if not i % 10:
                # debug("run=%2d i=%8d best=%d n_best=%d"%(r, i, best[r][i], sum([o==best[r][i] for o in obj])))
            # Assign fitness
            fitness = ranking(obj, SP)
            # Sample from population
            ix = sus(fitness)
            # Select offspring
            offspring = [pop[i].copy() for i in ix]
            # Apply mutation (in place)
            mutate(offspring, Pm)
            # Unconditional generational replacement
            pop = offspring
            i = i+1

        best[r].append(max([s.get_objective_value() for s in pop]))
        # debug("run=%2d i=%8d best=%d n_best=%d" % (r, i, best[r][i], sum([o==best[r][i] for o in obj])))
    return best_solution
