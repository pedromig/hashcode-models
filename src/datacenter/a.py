from __future__ import annotations

from typing import Optional, Iterable
from dataclasses import dataclass

import sys
import random
import copy

import nasf4nio as nio
from nasf4nio.tests import objective_increment_add_test
from nasf4nio.utils import non_repeating_lcg

@nio.property_test(objective_increment_add_test)
class Problem:    
    def __init__(self: Problem, r: int, s: int, u: int, p: int, m: int,
                 unavailable: tuple[tuple[int, int]], 
                 servers: tuple[tuple[int, int]]) -> None: 
        # Instance Parameters
        self.r, self.s, self.u, self.p, self.m = r, s, u, p, m
        self.unavailable = unavailable
        self.servers = servers

        # Setup
        self.__init_problem()

        print(f"Pools: {self.p}")
        print(f"Segments: {len(self.segments)}")

    def empty_solution(self: Problem) -> Solution:
        return Solution(self, init_ub=True)
    
    def heuristic_solution(self: Problem) -> Solution:
        solution = self.empty_solution()
        while (c := solution.heuristic_add_move()) is not None:
            solution.add(c)
        return solution

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
                               key=lambda x: (self.servers[x][0] /
                                              self.servers[x][1],
                                              self.servers[x][0]))

        # Sorted Servers (Inverse Permutation)
        self.pservers = [0] * self.m
        for i in range(self.m):
            self.pservers[self.sservers[i]] = i

    def __str__(self: Problem) -> str:
        s = f"{self.r} {self.s} {self.u} {self.p} {self.m}\n"
        s += "\n".join(f"{row} {slot}" for row, slot in self.unavailable) + "\n"
        s += "\n".join(f"{size} {capacity}" for size, capacity in self.servers)
        return s

class Solution: 
    def __init__(self: Solution, problem: Problem, used: Optional[set[int]] = None,
        unused: Optional[set[int]] = None, forbidden: Optional[set[int]] = None,
        alloc: Optional[list[tuple[int, int]]]= None,
        ss: Optional[list[int]] = None, gc: Optional[list[int]] = None, 
        cp: Optional[list[int]] = None, rc: Optional[list[list[int]]] = None,
        mc: Optional[list[int]] = None, objv: Optional[int] = None, 
        ub: Optional[float] = None, ub_kp: Optional[list[int]] = None, 
        ub_lim: Optional[list[int]] = None, ub_full: Optional[list[int]] = None, 
        ub_frac: Optional[list[int]] = None, iserver: Optional[int] = None, 
        init_ub: Optional[bool] = False) -> None: 

        # Problem
        self.problem = problem
        
        # Solution 
        self.iserver = 0 if iserver is None else iserver
        
        self.used = set() if used is None else used
        self.unused = set(range(self.problem.m)) if unused is None else unused 
        self.forbidden = set() if forbidden is None else forbidden
       
        # Assignment 
        self.alloc = [None] * self.problem.m if alloc is None else alloc

        # Capacities and Sizes
        self.ss = ([self.problem.segments[i][2] for i in range(len(self.problem.segments))] if ss is None else ss)
        self.gc = [0] * self.problem.p if gc is None else gc
        self.cp = [0] * self.problem.p if cp is None else cp
        self.rc = ([[0] * len(self.problem.rows) for _ in range(self.problem.p)] if rc is None else rc)
        self.mc = [0] * self.problem.p if mc is None else mc

        # Objective Value
        self.objv = 0 if objv is None else objv

        # Upper Bound
        if ub is None and init_ub:
            self.ub, self.ub_kp, self.ub_lim, self.ub_full, self.ub_frac = self.__init_upper_bound()
        else:
            self.ub, self.ub_kp, self.ub_lim, self.ub_full, self.ub_frac = ub, ub_kp, ub_lim, ub_full, ub_frac,
            
    def copy(self: Solution) -> Solution:
        return Solution(self.problem, self.used.copy(), self.unused.copy(), self.forbidden.copy(),
            self.alloc.copy(), self.ss.copy(), self.gc.copy(), self.cp.copy(),
            copy.deepcopy(self.rc), self.mc.copy(), self.objv, copy.copy(self.ub),
            self.ub_kp.copy(), self.ub_lim.copy(), self.ub_full.copy(), self.ub_frac.copy(),
            self.iserver)

    def feasible(self: Solution) -> bool:
        return True
    
    def score(self: Solution) -> int:
        return self.objv 

    def objective(self: Solution) -> int:
        return self.objv

    def upper_bound(self: Solution) -> float:
        return self.ub
    
    def components(self: Solution) -> Iterable[Component]:
        for server in self.used:
            yield Component(server, *self.alloc[server])
    
    def add_moves(self: Solution) -> Iterable[Component]:
        def standard() -> Iterable[Component]:
            for server in self.unused:
                for pool in range(self.problem.p):
                    for row in range(self.problem.r):
                        for segment in self.problem.rows[row]:
                            if self.__fits(server, segment):
                                yield Component(server, pool, segment)
                yield Component(server)
        
        def sequence() -> Iterable[Component]:
            if self.iserver < self.problem.m:
                server = self.problem.sservers[self.iserver]
                if server in self.unused:
                    for pool in range(self.problem.p):
                        for row in range(self.problem.r):
                            for segment in self.problem.rows[row]:
                                if self.__fits(server, segment):
                                    yield Component(server, pool, segment)
                    yield Component(server)
        yield from sequence()
            
    def heuristic_add_moves(self: Solution) -> Iterable[Component]:
        for server in self.problem.sservers:
            if server in self.unused:
                for pool in sorted(range(self.problem.p), key=lambda x: self.gc[x]):
                    for row in sorted(range(self.problem.r), key=lambda x: self.rc[pool][x]):
                        for segment in self.problem.rows[row]:
                            if self.__fits(server, segment):
                                yield Component(server, pool, segment)
                yield Component(server)
            
    def remove_moves(self: Solution) -> Iterable[Component]:
        for server in self.unused:
            pool, segment = self.alloc[server]
            yield Component(server, pool, segment)

    def local_moves(self: Solution) -> Iterable[LocalMove]:
        # Add (unused) Servers
        for add in self.unused:
            for r in range(self.problem.r):
                for segment in self.problem.rows[r]:
                    if self.__fits(add, segment):
                        for p in range(self.problem.p):
                            yield LocalMove((add, p, segment), None)

        # Remove (used) Servers
        for remove in self.used:
            pool, segment = self.alloc[remove]
            yield LocalMove(None, (remove, pool, segment))

        # Change (used) Server Segment
        for server in self.used:
            pool, segment = self.alloc[server]
            for r in range(self.problem.r):
                for seg in self.problem.rows[r]:
                    if seg != segment and self.__fits(server, seg):
                        yield LocalMove((server, pool, seg), (server, pool, segment))

        # Change (used) Server Pool
        for server in self.used:
            pool, segment = self.alloc[server]
            for p in range(self.problem.p):
                if p != pool:
                    yield LocalMove((server, p, segment), (server, pool, segment))

        # Swap (used) servers pools/segments
        for i in self.used:
            for j in self.used:
                if i != j:
                    pi, si = self.alloc[i]
                    pj, sj = self.alloc[j]
                    if si != sj and self.__swappable(i, j):
                        yield LocalMove((i, pi, si), (j, pj, sj), swap_segment=True)
                    if pi != pj:
                        yield LocalMove((i, pi, si), (j, pj, sj), swap_pool=True)

    def random_local_moves_wor(self: Solution) -> Iterable[LocalMove]:
        used = list(self.used)
        unused = list(self.unused)

        add_moves = len(self.unused) * len(self.problem.segments) * self.problem.p
        remove_moves = len(self.used)
        change_segment_moves = len(self.used) * len(self.problem.segments)
        change_pool_moves = len(self.used) * self.problem.p
        swap_moves = 2 * (len(used) ** 2)

        for move in non_repeating_lcg(add_moves + remove_moves + change_segment_moves + change_pool_moves + swap_moves):
            # Add (unused) Servers
            if move < add_moves:
                # Decode Server
                server = unused[move // (len(self.problem.segments) * self.problem.p)]
                move = move % (len(self.problem.segments) * self.problem.p)

                # Decode Segment
                segment = move // self.problem.p

                # Decode Pool
                pool = move % self.problem.p
                if self.__fits(server, segment):
                    yield LocalMove((server, pool, segment), None)
                continue
            move -= add_moves

            # Remove (used) Servers
            if move < remove_moves:
                # Decode Server
                server = used[move]

                # Fetch Pool and Segment
                pool, segment = self.alloc[server]

                # Local Move
                yield LocalMove(None, (server, pool, segment))
                continue
            move -= remove_moves

            # Change (used) Server Segment
            if move < change_segment_moves:
                # Decode Server
                server = used[move // len(self.problem.segments)]

                # Decode New Segment
                seg = move % len(self.problem.segments)

                # Server Current Pool and Segment
                pool, segment = self.alloc[server]

                # Local Move
                if seg != segment and self.__fits(server, seg):
                    yield LocalMove((server, pool, seg), (server, pool, segment))
                continue
            move -= change_segment_moves

            # Change (used) Server Pool
            if move < change_pool_moves:
                # Decode Server
                server = used[move // self.problem.p]

                # Decode New Pool
                p = move % self.problem.p

                # Server Current Pool and Segment
                pool, segment = self.alloc[server]

                # Local Move
                if p != pool:
                    yield LocalMove((server, p, segment), (server, pool, segment))
                continue
            move -= change_pool_moves

            # Swap (used) servers pools/segments

            # Decode server "i"
            i = used[move // (2 * len(used))]
            move = move % (2 * len(used))

            # Decode server "j"
            j = used[move // 2]

            # Decode move type
            move = move % 2

            pi, si = self.alloc[i]
            pj, sj = self.alloc[j]
            if move:
                if si != sj and self.__swappable(i, j):
                    yield LocalMove((i, pi, si), (j, pj, sj), swap_segment=True)
            else:
                if pi != pj:
                    yield LocalMove((i, pi, si), (j, pj, sj), swap_pool=True)
                    
    def heuristic_add_move(self: Solution) -> Component:
        return next(self.heuristic_add_moves(), None)

    def random_add_move(self: Solution) -> Component:
        unused = list(self.unused)
        server = random.choice(unused)
        pools = list(range(self.problem.p))
        for server in unused:
            rows = list(range(self.problem.r))
            row = random.choice(rows)
            for row in rows:
                segments = self.problem.rows[row].copy()
                random.shuffle(segments)
                for segment in segments:
                    pool = random.choice(pools)
                    if self.__fits(server, segment):
                        return Component(server, pool, segment)

    def random_remove_move(self: Solution) -> Component:
        used = list(self.used)
        random.shuffle(used)
        for server in used:
            pool, segment = self.alloc[server]
            return Component(server, pool, segment)

    def random_local_move(self: Solution) -> LocalMove:
        return next(self.random_local_moves_wor(), None)

    def add(self: Solution, c: Component) -> None:
        if self.__forbidden(c): 
            # Forbid Component
            self.__forbid(c)

            # Update Bound
            self.ub = self.__upper_bound_update_forbid(c, self.ub_kp, self.ub_lim, self.ub_full, self.ub_frac) 
        else: 
            # Add Component
            self.__add(c)

            # Update Objective Value
            self.objv = min(self.gc)

            # Update Bound
            self.ub = self.__upper_bound_update_add(c, self.ub_kp, self.ub_lim, self.ub_full, self.ub_frac)
            
        # Next Server
        self.iserver += 1 

    def remove(self: Solution, c: Component) -> None:
        # Remove Component
        self.__remove(c)

        # Update Objective Value
        self.objv = min(self.gc) 

        # Update Bound
        self.ub = self.__upper_bound_update_remove(c, self.ub_kp, self.ub_lim, self.ub_full, self.ub_frac)
        
        # Reset next server 
        self.iserver = 0
        
    def step(self: Solution, move: LocalMove) -> None:
        if move.swap_pool:
            self.__swap_pool(move.add, move.remove)
        elif move.swap_segment:
            self.__swap_segment(move.add, move.remove)
        else:
            if move.remove is not None:
                self.__remove(Component(*move.remove))

            if move.add is not None:
                self.__add(Component(*move.add))

        # Update Objective Value
        self.objv = min(self.gc)
        
        # Update Bound 
        self.ub = None

    def perturb(self: Solution, ks: int) -> None:
        for _ in range(ks):
            self.step(self.random_local_move())

    def objective_increment_add(self: Solution, c: Component) -> int:
        if self.__forbidden(c):
            return 0
        else:
            cp = self.cp[c.pool] + self.problem.servers[c.server][1]
            rc = self.rc[c.pool][self.problem.segments[c.segment][0]] + self.problem.servers[c.server][1]
            gc = self.gc.copy()
            gc[c.pool] = cp - max(self.mc[c.pool], rc)
            return min(gc) - self.objv

    def objective_increment_remove(self: Solution, c: Component) -> int:
        if self.__forbidden(c):
            return 0
        else:
            gc, rc = self.gc.copy(), self.rc.copy()
            cp = self.cp[c.pool] - self.problem.servers[c.server][1]  
            rc[c.pool][self.problem.segments[c.segment][0]] -= self.problem.servers[c.server][1]
            gc[c.pool] = cp - max(rc[c.pool])
            return min(gc) - self.objv
        
    def objective_increment_local(self: Solution, m: LocalMove) -> int:
        gc = self.gc.copy()
        if m.swap_segment or m.swap_pool:
            i, pi, si = m.add
            j, pj, sj = m.remove

            _, icap = self.problem.servers[i]
            _, jcap = self.problem.servers[j]

            ri, _, _ = self.problem.segments[si]
            rj, _, _ = self.problem.segments[sj]

            if m.swap_pool:
                cpi = self.cp[pi] - icap + jcap
                cpj = self.cp[pj] - jcap + icap

                rci = self.rc[pi].copy()
                rcj = self.rc[pj].copy()

                rci[ri] -= icap
                rcj[ri] += icap

                rci[rj] += jcap
                rcj[rj] -= jcap

                mci = max(rci)
                mcj = max(rcj)

            if m.swap_segment:
                cpi = self.cp[pi]
                cpj = self.cp[pj]

                rci = self.rc[pi].copy()
                rcj = self.rc[pj].copy()

                rci[ri] -= icap
                rcj[ri] += jcap

                rci[rj] += icap
                rcj[rj] -= jcap

                mci = max(rci)
                mcj = max(rcj)

            gc[pi] = cpi - mci
            gc[pj] = cpj - mcj
        else:
            if m.add is not None:
                server, pool, segment = m.add
                _, cap = self.problem.servers[server]
                row = self.problem.segments[segment][0]

                cp = self.cp[pool] + cap
                rc = self.rc[pool][row] + cap
                gc[pool] = cp - max(self.mc[pool], rc)

            if m.remove is not None:
                server, pool, segment = m.remove
                _, cap = self.problem.servers[server]
                row = self.problem.segments[segment][0]

                cp = self.cp[pool] - cap
                rc = self.rc[pool][row] - cap
                gc[pool] = cp - max(self.rc[pool][:row] + self.rc[pool][row + 1:] + [rc]) 
        return min(gc) - self.objv

    def upper_bound_increment_add(self: Solution, c: Component) -> float:
        if self.__forbidden(c): 
            ub = self.__upper_bound_update_forbid(c, self.ub_kp.copy(), self.ub_lim.copy(),
                                                  self.ub_full.copy(), self.ub_frac.copy())
        else:
            _, capacity = self.problem.servers[c.server]
            row, _, _ = self.problem.segments[c.segment]

            self.cp[c.pool] += capacity
            self.rc[c.pool][row] += capacity

            ub = self.__upper_bound_update_add(c, self.ub_kp.copy(), self.ub_lim.copy(), 
                                            self.ub_full.copy(), self.ub_frac.copy())
            
            self.cp[c.pool] -= capacity
            self.rc[c.pool][row] -= capacity
        return ub - self.ub

    def upper_bound_increment_remove(self: Solution, c: Component) -> float:
        _, capacity = self.problem.servers[c.server]
        row, _, _ = self.problem.segments[c.segment]

        self.cp[c.pool] -= capacity
        self.rc[c.pool][row] -= capacity

        ub = self.__upper_bound_update_remove(c, self.ub_kp.copy(), self.ub_lim.copy(),
                                              self.ub_full.copy(), self.ub_frac.copy())

        self.cp[c.pool] += capacity
        self.rc[c.pool][row] += capacity

        return ub - self.ub
    
    def __fits(self: Solution, server: int, segment: int) -> bool:
        return self.problem.servers[server][0] <= self.ss[segment]
     
    def __swappable(self: Solution, i: int, j: int) -> bool:
        return (
            self.ss[self.alloc[i][1]] - self.problem.servers[i][0] >= self.problem.servers[j][0]
            and self.ss[self.alloc[j][1]] - self.problem.servers[j][0] >= self.problem.servers[i][0]
        )
         
    def __forbid(self: Solution, c: Component) -> None:
        # Forbid Server
        self.unused.remove(c.server)
        self.forbidden.add(c.server)
        
    def __forbidden(self: Solution, c: Component) -> bool:
        return c.pool is None or c.segment is None

    def __add(self: Solution, c: Component) -> None:
        size, capacity = self.problem.servers[c.server]
        row, _, _ = self.problem.segments[c.segment]

        # Allocate Server
        self.used.add(c.server)
        if c.server in self.forbidden:
            self.forbidden.remove(c.server)
        else:
            self.unused.remove(c.server)
        self.alloc[c.server] = (c.pool, c.segment)

        # Update Remaining Segment Space
        self.ss[c.segment] -= size

        # Update Capacities
        self.cp[c.pool] += capacity
        self.rc[c.pool][row] += capacity
        self.mc[c.pool] = max(self.mc[c.pool], self.rc[c.pool][row])
        self.gc[c.pool] = self.cp[c.pool] - self.mc[c.pool]

    def __remove(self: Solution, c: Component) -> None:
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
         
    def __swap_pool(self: Solution, a: tuple[int, int, int], b: tuple[int, int, int]) -> None:
        i, pi, si = a
        j, pj, sj = b

        _, icap = self.problem.servers[i]
        _, jcap = self.problem.servers[j]

        ri, _, _ = self.problem.segments[si]
        rj, _, _ = self.problem.segments[sj]

        self.cp[pi] = self.cp[pi] - icap + jcap
        self.cp[pj] = self.cp[pj] - jcap + icap

        self.alloc[i] = (pj, si)
        self.alloc[j] = (pi, sj)

        self.rc[pi][ri] -= icap
        self.rc[pj][ri] += icap

        self.rc[pi][rj] += jcap
        self.rc[pj][rj] -= jcap

        self.mc[pi] = max(self.rc[pi])
        self.mc[pj] = max(self.rc[pj])

        self.gc[pi] = self.cp[pi] - self.mc[pi]
        self.gc[pj] = self.cp[pj] - self.mc[pj]

    def __swap_segment(self: Solution, a: tuple[int, int, int], b: tuple[int, int, int]) -> None:
        i, pi, si = a
        j, pj, sj = b

        isize, icap = self.problem.servers[i]
        jsize, jcap = self.problem.servers[j]

        ri, _, _ = self.problem.segments[si]
        rj, _, _ = self.problem.segments[sj]

        self.ss[si] = self.ss[si] - isize + jsize
        self.ss[sj] = self.ss[sj] - jsize + isize

        self.alloc[i] = (pi, sj)
        self.alloc[j] = (pj, si)

        self.rc[pi][ri] -= icap
        self.rc[pj][ri] += jcap

        self.rc[pi][rj] += icap
        self.rc[pj][rj] -= jcap

        self.mc[pi] = max(self.rc[pi])
        self.mc[pj] = max(self.rc[pj])

        self.gc[pi] = self.cp[pi] - self.mc[pi]
        self.gc[pj] = self.cp[pj] - self.mc[pj]
                    
    def __init_upper_bound(self: Solution) -> tuple[float, list[int], list[int], list[int], list[float]]:
        # Knapsacks
        knapsacks = [0] * self.problem.r
        for r, row in enumerate(self.problem.rows):
            for i in row:
                for j in range(self.problem.r):
                    knapsacks[j] += self.problem.segments[i][2] if r != j else 0

        # Bound Initialization
        ub = sys.maxsize
        kp, lim = [0] * self.problem.r, [0] * self.problem.r
        full, frac = [0] * self.problem.r, [0] * self.problem.r
        
        for i in range(self.problem.r):
            kp[i] = knapsacks[i]
            for j in self.problem.sservers:
                size, capacity = self.problem.servers[j]
                if size <= kp[i]:
                    kp[i] -= size
                    full[i] += capacity
                    lim[i] += 1
                else:
                    frac[i] = capacity * (kp[i] / size)
                    break
            ub = min(ub, (full[i] + frac[i]) / self.problem.p)
        return ub, kp, lim, full, frac

    def __upper_bound_update_add(self: Solution, c: Component, 
                                 ub_kp: list[int],  ub_lim: list[int],
                                 ub_full: list[int], ub_frac: list[int]) -> float:
        ub = sys.maxsize
        row, _, _ = self.problem.segments[c.segment]
        size, capacity = self.problem.servers[c.server]

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
            ub = min(ub, total / self.problem.p)
            ub = self.__row_upper_bound(i, total, ub)
        return ub

    def __upper_bound_update_remove(self: Solution, c: Component, 
                                    ub_kp: list[int], ub_lim: list[int],
                                    ub_full: list[int], ub_frac: list[int]) -> float:
        ub = sys.maxsize
        size, capacity = self.problem.servers[c.server]
        row, _, _ = self.problem.segments[c.segment]

        for i in range(self.problem.r):
            if i != row and self.problem.pservers[c.server] >= ub_lim[i]:
                ub_kp[i] += size
                ub_full[i] -= capacity
                sx = self.problem.sservers[ub_lim[i]]
                sz, cap = self.problem.servers[sx]
                while ub_lim[i] < self.problem.m:
                    sx = self.problem.sservers[ub_lim[i]]
                    sz, cap = self.problem.servers[sx]
                    if sx in self.unused:
                        if sz <= ub_kp[i]:
                            ub_kp[i] -= sz
                            ub_full[i] += cap
                        else:
                            ub_frac[i] = cap * (ub_kp[i] / sz)
                            break
                    ub_lim[i] += 1
            elif i == row and self.problem.pservers[c.server] <= ub_lim[i]:
                ub_kp[i] -= size
                ub_full[i] += capacity
                sx = self.problem.sservers[ub_lim[i]]
                sz, cap = self.problem.servers[sx]
                while ub_kp[i] < 0:
                    ub_lim[i] -= 1
                    sx = self.problem.sservers[ub_lim[i]]
                    sz, cap = self.problem.servers[sx]
                    if sx in self.unused:
                        ub_full[i] -= cap
                        ub_kp[i] += sz
                ub_frac[i] = cap * (ub_kp[i] / sz)
            total = ub_full[i] + ub_frac[i]
            ub = min(ub, total / self.problem.p)
            ub = self.__row_upper_bound(i, total, ub)
        return ub
    
    def __upper_bound_update_forbid(self: Solution, c: Component, 
                                    ub_kp: list[int], ub_lim: list[int],
                                    ub_full: list[int], ub_frac: list[int]) -> float:
        ub = sys.maxsize
        size, capacity = self.problem.servers[c.server]
        for i in range(self.problem.r):
            if self.problem.pservers[c.server] <= ub_lim[i]:
                if self.problem.pservers[c.server] < ub_lim[i]:
                    ub_kp[i] += size
                    ub_full[i] -= capacity
                ub_frac[i] = 0
                while ub_lim[i] < self.problem.m:
                    sx = self.problem.sservers[ub_lim[i]]
                    sz, cap = self.problem.servers[sx]
                    if (sx in self.unused and sx != c.server and sx not in self.forbidden):
                        if sz <= ub_kp[i]:
                            ub_kp[i] -= sz
                            ub_full[i] += cap
                        else:
                            ub_frac[i] = cap * (ub_kp[i] / sz)
                            break
                    ub_lim[i] += 1
            total = ub_full[i] + ub_frac[i] 
            ub = min(ub, total / self.problem.p)
            ub = self.__row_upper_bound(i, total, ub)
        return ub 

    def __row_upper_bound(self: Solution, row: int, total: float, ub: float) -> float:
        count = self.problem.p
        p, ignored = 0, set()
        while p < self.problem.p:
            cap = self.cp[p] - self.rc[p][row]
            if p not in ignored and cap > total / count:
                count -= 1
                total -= cap
                ub = min(ub, total / count)
                ignored.add(p)
                p = -1
            p += 1
        return ub 
    
    def __eq__(self: Solution, other: Solution) -> bool:
        for server in self.used:
            if server in other.used and self.alloc[server] == other.alloc[server]:
                continue
            else:
                return False
        return True                

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
    
@dataclass(order=True) 
class LocalMove:
    add: Optional[tuple[int]]
    remove: Optional[tuple[int]] 
    swap_segment: bool = False
    swap_pool: bool = False
    
@dataclass(order=True)
class Component: 
    server: int
    pool: Optional[int] = None
    segment: Optional[int] = None

    def id(self):
        return self.server, self.pool, self.segment
