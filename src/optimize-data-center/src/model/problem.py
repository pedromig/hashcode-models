from __future__ import annotations

from .solution import Solution
from .component import Component

import sys
import random
 
class Problem:
    def __init__(self: Problem, r: int, s: int, u: int, p: int, m: int, 
                unavailable: tuple[tuple], servers: tuple[tuple]) -> None: 
        # Instance Parameters
        self.r, self.s, self.u, self.p, self.m = r, s, u, p, m     
        self.unavailable = unavailable
        self.servers = servers
        
        # Setup
        self.__init_problem() 
        self.__init_upper_bound()

    def empty_solution(self: Problem) -> Solution:  
        return Solution(self, None, None, None, None, None, None, None, None,
                        None, self.__ub, self.__ub_kp, self.__ub_lim,
                        self.__ub_full, self.__ub_frac)

    def random_solution(self: Problem) -> Solution:
        solution = self.empty_solution()
        while (c := solution.random_add_move()) is not None:
            solution.add(c)
        return solution
              
    @staticmethod
    def from_stdin() -> Problem:
        # Instance Parameters
        r, s, u, p, m = map(int, input().split())

        # Unavailable Slots
        unavailable = [] 
        for _ in range(u):
            row, slot = map(int, input().split())
            unavailable.append((row, slot)) 

        # Servers
        servers = []
        for _ in range(m):
            size, capacity = map(int, input().split())
            servers.append((size, capacity))
            
        return Problem(r, s, u, p, m, tuple(unavailable), tuple(servers))
     
    def __init_problem(self: Problem) -> None:  
        # Preprocess
        slots = [{-1, self.s} for _ in range(self.r)] 
        for row, slot in self.unavailable:    
            slots[row].add(slot)
            
        # Segments / Rows
        segments, rows = [], []
        for r in range(self.r):
            u, row = sorted(slots[r]), []
            for i in range(len(u) - 1):
                x, y = u[i] + 1, u[i + 1] - 1 
                if x < y:
                    segments.append((r, x, y - x + 1))
                    row.append(len(segments) - 1)
            rows.append(row)
        self.rows = rows   
        self.segments = segments           

        # Sorted Servers
        self.sservers = sorted(range(self.m), 
                               key = lambda x: (self.servers[x][0] /
                                                self.servers[x][1],
                                                self.servers[x][0]))

        # Sorted Servers (Inverse Permutation)
        self.pservers = [0] * self.m
        for i in range(self.m):
            self.pservers[self.sservers[i]] = i
        
          
    def __init_upper_bound(self: Problem) -> None:
        # Knapsacks 
        knapsacks = [0] * self.r
        for r, row in enumerate(self.rows): 
            for i in row: 
                for j in range(self.r):
                    knapsacks[j] += self.segments[i][2] if r != j else 0 
              
        # Bound Initialization
        ub = sys.maxsize 
        kp, lim = [0] * self.r, [0] * self.r
        full, frac = [0] * self.r, [0] * self.r
  
        for i in range(self.r):
            kp[i] = knapsacks[i]
            for j in self.sservers:
                size, capacity = self.servers[j]
                if size <= kp[i]: 
                    kp[i] -= size
                    full[i] += capacity 
                    lim[i] += 1 
                else:
                    frac[i] = capacity  * (kp[i] / size)
                    break
            ub = min(ub , (full[i] + frac[i]) / self.p)
               
        self.__ub = ub
        self.__ub_kp = kp
        self.__ub_lim = lim
        self.__ub_full = full
        self.__ub_frac = frac
         
    def __str__(self: Problem) -> str:     
        s = f"{self.r} {self.s} {self.u} {self.p} {self.m}\n"
        s += "\n".join(f"{row} {slot}" for row, slot in self.unavailable) + "\n"
        s += "\n".join(f"{size} {capacity}" for size, capacity in self.servers)  
        return s
    
    def __repr__(self: Problem) -> str: 
        s = f"r = {self.r}\n" 
        s += f"s = {self.s}\n"
        s += f"u = {self.u}\n"
        s += f"p = {self.p}\n"
        s += f"m = {self.m}\n"
        s += f"unavailable = {self.unavailable}\n"
        s += f"servers = {self.servers}\n"
        s += f"segments = {self.segments}\n"
        s += f"rows = {self.rows}\n"
        s += f"sservers = {self.sservers}\n"
        s += f"pservers = {self.pservers}\n" 
        s += f"ub_kp = {__self.ub_kp}\n"
        s += f"ub_full = {self.__ub_full}\n"
        s += f"ub_frac = {self.__ub_frac}\n"
        s += f"ub_lim = {self.__ub_lim}\n"
        return s
     
