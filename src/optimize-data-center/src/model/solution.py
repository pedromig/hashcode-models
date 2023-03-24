from __future__ import annotations
from typing import TYPE_CHECKING, Iterable 
    
from .component import *
from .move import *

if TYPE_CHECKING:
    from .problem import *

import sys
import random
import operator

# TODO: Documentation

class Solution:
    def __init__(self: Solution, problem: Problem, used: set[int] | None, 
                 unused: set[int] | None, alloc: list[tuple[int]] | None, 
                 ss: list[int] | None, gc: list[int] | None, cp: list[int] | None, 
                 rc: list[int] | None, mc: list[int] | None, objv: tuple[int] | None, ub: float, 
                 ub_kp: list[int], ub_lim: list[int], ub_full: list[int], ub_frac: list[int]) -> None: 
        # Problem
        self.problem = problem 
       
        # Solution 
        self.used = set() if used is None else used 
        self.unused = set(range(self.problem.m)) if unused is None else unused
        self.alloc = [None] * self.problem.m if alloc is None else alloc
         
        # Capacities and Sizes
        self.ss = [self.problem.segments[i][2] for i in range(len(self.problem.segments))] if ss is None else ss
        self.gc = [0] * self.problem.p if gc is None else gc
        self.cp = [0] * self.problem.p if cp is None else cp
        self.rc = [[0] * len(self.problem.rows) for _ in range(self.problem.p)] if rc is None else rc
        self.mc = [0] * self.problem.p if mc is None else mc

        # Objective Value
        self.objv = tuple(sorted(self.gc)) if objv is None else objv
                
        # Upper Bound 
        self.ub = 0 if ub is None else ub
        self.ub_kp = ub_kp
        self.ub_lim = ub_lim 
        self.ub_full = ub_full
        self.ub_frac = ub_frac 
  
    def copy(self: Solution) -> Solution:
        return Solution(self.problem, self.used.copy(), self.unused.copy(),
                        self.alloc.copy(), self.ss.copy(), self.gc.copy(),
                        self.cp.copy(), self.rc.copy(), self.mc.copy(),
                        self.objv, self.ub, self.ub_kp.copy(),
                        self.ub_lim.copy(), self.ub_full.copy(),
                        self.ub_frac.copy())
        
    def feasible(self: Solution) -> bool:
        return True 
    
    def score(self: Solution) -> int:
        return min(self.gc) 

    def objective_value(self: Solution) -> tuple[int]: 
        return self.objv
 
    def upper_bound(self: Solution) -> float:
        return self.ub
  
    def enum_add_move(self: Solution) -> Iterable[Component]:
        for server in self.unused:
            for pool in range(self.problem.p):
                for row in range(self.problem.r):
                    for segment in self.problem.rows[row]: 
                        if self.__fits(server, segment):                  
                             yield Component(self.problem, server, pool, segment)

                             
    def enum_heuristic_add_move(self: Solution) -> Iterable[Component]:
        for server in self.problem.sservers:
            if server in self.unused:
                for pool in sorted(range(self.problem.p), key = lambda x: self.gc[x]):
                    for row in sorted(range(self.problem.r), key = lambda x: self.rc[pool][x]):
                        for segment in self.problem.rows[row]:
                            if self.__fits(server, segment):
                                yield Component(self.problem, server, pool, segment)
            yield Component(self.problem, server, None, None) 
                                
    def enum_local_move(self: Solution) -> Iterable[LocalMove]:  
        # Add (unused) Servers 
        for add in self.unused:
            for r in range(self.problem.r):
                for segment in self.problem.rows[r]: 
                    if self.__fits(add, segment):
                        for p in range(self.problem.p):
                            yield LocalMove(self.problem, (add, p, segment), None)
   
        # # Remove (used) Servers
        for remove in self.used:
            yield LocalMove(self.problem, None, (remove, *self.alloc[remove]))

        # # Swap (used) servers
        for i in self.used:
            for j in self.used:
                if i != j :
                    if self.__swappable(i, j):
                        a, b = (i, *self.alloc[i]), (j, *self.alloc[j])
                        yield LocalMove(self.problem, a, b, swap = True)
                        yield LocalMove(self.problem, a, b, swap = True, pool = True)    

        # Swap (used) server by another (unused) server
        # TODO: Test all pools
        ...

    def random_add_move(self: Solution) -> Component: 
        unused = list(self.unused)
        random.shuffle(unused)
        pools = list(range(self.problem.p))
        for server in unused:
            rows = list(range(self.problem.r))
            random.shuffle(rows)
            for row in rows:
                segments = self.problem.rows[row].copy()
                random.shuffle(segments)
                for segment in segments:
                    pool = random.choice(pools)
                    if self.__fits(server, segment):
                        return Component(self.problem, server, pool, segment) 
    
    def random_remove_move(self: Solution) -> Component:
        used = list(self.used)
        random.shuffle(used)
        for server in used:
            pool, segment = self.alloc[server]
            return Component(self.problem, server, pool, segment)

    def random_local_move(self: Solution) -> LocalMove:
        moves = list(self.enum_local_move())
        random.shuffle(moves)
        for move in moves:
            return move

    def random_local_move_wor(self: Solution) -> Iterable[LocalMove]:
        moves = list(self.enum_local_move())
        random.shuffle(moves)
        for move in moves:
            yield move
 
    def step(self: Solution, move: LocalMove) -> None:
        if move.swap:
            ...

        if move.add is not None:
            self.add(Component(self.problem, *move.add))

        if move.remove is not None:
            self.remove(Component(self.problem, *move.remove))
      
    def add(self: Solution, c: Component) -> None:
        # Add Component 
        self.__addServer(c)
        
        # Update Score
        self.objv = tuple(sorted(self.gc))
        
        # Update Bound 
        self.ub = self.__bound_update(c, self.ub_kp, self.ub_lim, self.ub_full, self.ub_frac)
        
    def remove(self: Solution, c: Component) -> None:
        # Remove Component
        self.__removeServer(c)
          
        # Update Score
        self.objv = tuple(sorted(self.gc))

        # Update Bound 
        self.ub = self.__bound_update(c, self.ub_kp, self.ub_lim, self.ub_full, self.ub_frac)

    def objective_increment_local(self: Solution) -> int:
        ...
    
    def objective_increment_add(self: Solution, c: Component) -> int:
        cp = self.cp[c.pool] + self.problem.servers[c.server][1]
        rc = self.rc[c.pool][self.problem.segments[c.segment][0]] + self.problem.servers[c.server][1]
        gc = self.gc.copy()
        gc[c.pool] = cp - max(self.mc[c.pool], rc) 
        gc.sort()
        new_objv = tuple(gc)

        return tuple(map(operator.sub, new_objv, self.objv))
    
    def upper_bound_increment_add(self: Solution, c: Component) -> float:  
        ub = self.__bound_update(c, self.ub_kp.copy(), self.ub_lim.copy(),
                                 self.ub_full.copy(), self.ub_frac.copy())   
        return ub - self.ub
          
    def __fits(self: Solution, server: int, segment: int) -> bool:
        return self.problem.servers[server][0] <= self.ss[segment]        
 
    def __swappable(self: Solution, i: int, j: int, inplace = False) -> bool:
        iseg, jseg = self.alloc[i][1], self.alloc[j][1]
        isize, _ = self.problem.servers[i]
        jsize, _ = self.problem.servers[j]

        # Conditions
        a = self.ss[iseg] + isize - jsize >= 0 
        b = self.ss[jseg] + jsize - isize >= 0
        return b if inplace is False else a and b
    
    def __addServer(self: Solution, c: Component):
        size, capacity = self.problem.servers[c.server]  
        row, _, _ = self.problem.segments[c.segment]  
        
        # Allocate Server
        self.used.add(c.server)
        self.unused.remove(c.server) 
        self.alloc[c.server] = (c.pool, c.segment) 
        
        # Update Remaining Segment Space
        self.ss[c.segment] -= size  
        
        # Update Capacities
        self.cp[c.pool] += capacity 
        self.rc[c.pool][row] += capacity  
        self.mc[c.pool] = max(self.mc[c.pool], self.rc[c.pool][row])
        self.gc[c.pool] = self.cp[c.pool] - self.mc[c.pool] 
     
    def __removeServer(self: Solution, c: Component) -> None: 
        size, capacity = self.problem.servers[c.server]  
        row, _, _ = self.problem.segments[c.segment]  
        
        # Deallocate Server
        self.used.remove(c.server)
        self.unused.add(c.server) 
        self.alloc[c.server] = None 
        
        # Update Remaining Segment Space
        self.ss[c.segment] += size  
     
        # Update Capacities 
        self.rc[c.pool][row] -= capacity
        self.cp[c.pool] -= capacity
        self.mc[c.pool] = max(self.rc[c.pool])
        self.gc[c.pool] = self.cp[c.pool] - self.mc[c.pool] 
 
    def __bound_update(self: Solution, c: Component, ub_kp: list[int],
                           ub_lim: list[int], ub_full: list[int], 
                           ub_frac: list[int]) -> float:
        ub = sys.maxsize 
        row, _ , _ = self.problem.segments[c.segment] 
        size, capacity = self.problem.servers[c.server] 
 
        # Incremental Bound
        for i in range(self.problem.r): 
            if i != row and self.problem.pservers[c.server] >= ub_lim[i]:
                ub_kp[i] -= size 
                ub_full[i] += capacity 
                sx = self.problem.sservers[ub_lim[i]]  
                sz, cap = self.problem.servers[sx] 
                while ub_kp[i] < 0:
                    ub_lim[i] -= 1 
                    sx = self.problem.sservers[ub_lim[i]] 
                    sz, cap = self.problem.servers[sx]
                    if sx in self.unused:
                        ub_kp[i] += sz
                        ub_full[i] -= cap
                ub_frac[i] = cap * (ub_kp[i] / sz) 
            elif i == row and self.problem.pservers[c.server] <= ub_lim[i]: 
                if self.problem.pservers[c.server] < ub_lim[i]: 
                    ub_kp[i] += size 
                    ub_full[i] -= capacity 
                ub_frac[i] = 0
                while ub_lim[i] < self.problem.m:   
                    sx = self.problem.sservers[ub_lim[i]] 
                    sz, cap = self.problem.servers[sx] 
                    if sx in self.unused and sx != c.server:
                        if sz <= ub_kp[i]:
                            ub_kp[i] -= sz
                            ub_full[i] += cap
                        else:
                            ub_frac[i] = cap * (ub_kp[i] / sz)
                            break
                    ub_lim[i] += 1
            total = ub_full[i] + ub_frac[i] 
            count = self.problem.p 
            ub = min(ub, total / count)
        
            # Correction
            p, ignored = 0, set()
            while p < self.problem.p:
                cap = self.cp[p] - self.rc[p][i] 
                if p not in ignored and cap > total / count:
                    count -= 1
                    total -= cap
                    ub = min(ub , total / count) 
                    ignored.add(p)
                    p = -1
                p += 1
        return ub 
                   
    def __str__(self: Solution) -> str: 
        s, slots = "", [self.problem.segments[i][1] for i in range(len(self.problem.segments))]
        for server, allocation in enumerate(self.alloc):
            if allocation is not None:
                pool, segment = allocation
                row = self.problem.segments[segment][0]
                s += f"{row} {slots[segment]} {pool}\n"
                slots[segment] += self.problem.servers[server][0]
            else:
                s += "x\n"
        return s[:-1]
   
    def __repr__(self: Solution) -> str: 
        s = f"unused = {self.unused}\n"
        s += f"alloc = {self.alloc}\n"
        s += f"ss = {self.ss}\n"
        s += f"gc = {self.gc}\n"
        s += f"cp = {self.cp}\n"
        s += f"rc = {self.rc}\n"
        s += f"ub_kp = {self.ub_kp}\n"
        s += f"ub_lim = {self.ub_lim}\n"
        s += f"ub_full = {self.ub_full}\n"
        s += f"ub_frac = {self.ub_frac}\n"
        s += f"objv = {self.objv}\n"
        s += f"ub = {self.ub}"
        return s
