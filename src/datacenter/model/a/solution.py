from __future__ import annotations
from typing import TYPE_CHECKING, Iterable

from .component import *
from .move import *

if TYPE_CHECKING:
    from .problem import *

import sys
import random
import operator
import math
import copy


class Solution:
    def __init__(
        self: Solution,
        problem: Problem,
        used: set[int] | None = None,
        unused: set[int] | None = None,
        alloc: list[tuple[int]] | None = None,
        ss: list[int] | None = None,
        gc: list[int] | None = None,
        cp: list[int] | None = None,
        rc: list[list[int]] | None = None,
        mc: list[int] | None = None,
        objv: tuple[int] | None = None,
        ub: float | None = None,
        ub_kp: list[int] | None = None,
        ub_lim: list[int] | None = None,
        ub_full: list[int] | None = None,
        ub_frac: list[int] | None = None,
        init_ub: bool = False,
    ) -> None: 
        # Problem
        self.problem = problem

        # Solution
        self.used = set() if used is None else used
        self.unused = set(range(self.problem.m)) if unused is None else unused
        self.alloc = [None] * self.problem.m if alloc is None else alloc

        # Capacities and Sizes
        self.ss = (
            [self.problem.segments[i][2]
                for i in range(len(self.problem.segments))]
            if ss is None
            else ss
        )
        self.gc = [0] * self.problem.p if gc is None else gc
        self.cp = [0] * self.problem.p if cp is None else cp
        self.rc = (
            [[0] * len(self.problem.rows) for _ in range(self.problem.p)]
            if rc is None
            else rc
        )
        self.mc = [0] * self.problem.p if mc is None else mc

        # Objective Value
        self.objv = tuple(sorted(self.gc)) if objv is None else objv

        # Upper Bound
        if ub is None and init_ub:
            (
                self.ub,
                self.ub_kp,
                self.ub_lim,
                self.ub_full,
                self.ub_frac,
            ) = self.__init_bound()
        else:
            self.ub, self.ub_kp, self.ub_lim, self.ub_full, self.ub_frac = (
                ub,
                ub_kp,
                ub_lim,
                ub_full,
                ub_frac,
            )

    def copy(self: Solution) -> Solution:
        return Solution(
            self.problem,
            self.used.copy(),
            self.unused.copy(),
            self.alloc.copy(),
            self.ss.copy(),
            self.gc.copy(),
            self.cp.copy(),
            copy.deepcopy(self.rc),
            self.mc.copy(),
            self.objv,
            copy.copy(self.ub),
            self.ub_kp.copy(),
            self.ub_lim.copy(),
            self.ub_full.copy(),
            self.ub_frac.copy(),
        )

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
                for pool in sorted(range(self.problem.p), key=lambda x: self.gc[x]):
                    for row in sorted(
                        range(self.problem.r), key=lambda x: self.rc[pool][x]
                    ):
                        for segment in self.problem.rows[row]:
                            if self.__fits(server, segment):
                                yield Component(self.problem, server, pool, segment)

    def enum_remove_move(self: Solution) -> Iterable[Component]:
        for server in self.unused:
            pool, segment = self.alloc[server]
            yield Component(problem, server, pool, segment)

    def enum_local_move(self: Solution):
        return self.__enum_local_move_lcg()

    def enum_random_local_move_wor(self: Solution) -> Iterable[LocalMove]:
        moves = list(self.enum_local_move())
        random.shuffle(moves)
        for move in moves:
            yield move

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

    def step(self: Solution, move: LocalMove) -> None:
        if move.swap_pool:
            self.__swap_pool(move.add, move.remove)
        elif move.swap_segment:
            self.__swap_segment(move.add, move.remove)
        else:
            if move.remove is not None:
                self.__remove(Component(self.problem, *move.remove))

            if move.add is not None:
                self.__add(Component(self.problem, *move.add))

        # Update Objective Value
        self.objv = tuple(sorted(self.gc))

        self.ub = None

    def perturb(self: Solution, kick) -> None:
        for _ in range(kick):
            self.step(self.random_local_move())

    def add(self: Solution, c: Component) -> None:
        # Add Component
        self.__add(c)

        # Update Objective Value
        self.objv = tuple(sorted(self.gc))

        # Update Bound
        self.ub = self.__upper_bound_update_add(
            c, self.ub_kp, self.ub_lim, self.ub_full, self.ub_frac
        )

    def remove(self: Solution, c: Component) -> None:
        # Remove Component
        self.__remove(c)

        # Update Objective Value
        self.objv = tuple(sorted(self.gc))

        # Update Bound
        self.ub = self.__upper_bound_update_remove(
            c, self.ub_kp, self.ub_lim, self.ub_full, self.ub_frac
        )

    def objective_increment_add(self: Solution, c: Component) -> int:
        cp = self.cp[c.pool] + self.problem.servers[c.server][1]
        rc = (
            self.rc[c.pool][self.problem.segments[c.segment][0]]
            + self.problem.servers[c.server][1]
        )
        gc = self.gc.copy()
        gc[c.pool] = cp - max(self.mc[c.pool], rc)
        gc.sort()
        return tuple(map(operator.sub, tuple(gc), self.objv))

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
                gc[pool] = cp - max(
                    self.rc[pool][:row] + self.rc[pool][row + 1:] + [rc]
                )
        gc.sort()
        return tuple(map(operator.sub, tuple(gc), self.objv))

    def upper_bound_increment_add(self: Solution, c: Component) -> float:

        _, capacity = self.problem.servers[c.server]
        row, _, _ = self.problem.segments[c.segment]

        self.cp[c.pool] += capacity
        self.rc[c.pool][row] += capacity

        ub = self.__upper_bound_update_add(
            c,
            self.ub_kp.copy(),
            self.ub_lim.copy(),
            self.ub_full.copy(),
            self.ub_frac.copy(),
        )

        self.cp[c.pool] -= capacity
        self.rc[c.pool][row] -= capacity
        return ub - self.ub

    def upper_bound_increment_remove(self: Solution, c: Component) -> float:
        _, capacity = self.problem.servers[c.server]
        row, _, _ = self.problem.segments[c.segment]

        self.cp[c.pool] -= capacity
        self.rc[c.pool][row] -= capacity

        ub = self.__upper_bound_update_remove(
            c,
            self.ub_kp.copy(),
            self.ub_lim.copy(),
            self.ub_full.copy(),
            self.ub_frac.copy(),
        )

        self.cp[c.pool] += capacity
        self.rc[c.pool][row] += capacity

        return ub - self.ub

    def __fits(self: Solution, server: int, segment: int) -> bool:
        return self.problem.servers[server][0] <= self.ss[segment]
     
    def __swappable(self: Solution, i: int, j: int) -> bool:
        return (
            self.ss[self.alloc[i][1]] - self.problem.servers[i][0]
            >= self.problem.servers[j][0]
            and self.ss[self.alloc[j][1]] - self.problem.servers[j][0]
            >= self.problem.servers[i][0]
        )

    def __add(self: Solution, c: Component):
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
         
    def __swap_pool(self: Solution, a: tuple[int], b: tuple[int]) -> None:
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

    def __swap_segment(self: Solution, a: tuple[int], b: tuple[int]) -> None:
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

    def __enum_local_move_lcg(self: Solution) -> Iterable[LocalMove]:
        used = list(self.used)
        unused = list(self.unused)

        add_moves = len(self.unused) * \
            len(self.problem.segments) * self.problem.p
        remove_moves = len(self.used)
        change_segment_moves = len(self.used) * len(self.problem.segments)
        change_pool_moves = len(self.used) * self.problem.p
        swap_moves = 2 * (len(used) ** 2)

        for move in self.__non_repeating_lcg(
            add_moves
            + remove_moves
            + change_segment_moves
            + change_pool_moves
            + swap_moves
        ):
            # Add (unused) Servers
            if move < add_moves:
                # Decode Server
                server = unused[move //
                                (len(self.problem.segments) * self.problem.p)]
                move = move % (len(self.problem.segments) * self.problem.p)

                # Decode Segment
                segment = move // self.problem.p

                # Decode Pool
                pool = move % self.problem.p
                if self.__fits(server, segment):
                    yield LocalMove(self.problem, (server, pool, segment), None)
                continue
            move -= add_moves

            # Remove (used) Servers
            if move < remove_moves:
                # Decode Server
                server = used[move]

                # Fetch Pool and Segment
                pool, segment = self.alloc[server]

                # Local Move
                yield LocalMove(self.problem, None, (server, pool, segment))
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
                    yield LocalMove(
                        self.problem, (server, pool,
                                       seg), (server, pool, segment)
                    )
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
                    yield LocalMove(
                        self.problem, (server, p,
                                       segment), (server, pool, segment)
                    )
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
                    yield LocalMove(
                        self.problem, (i, pi, si), (j, pj,
                                                    sj), swap_segment=True
                    )
            else:
                if pi != pj:
                    yield LocalMove(
                        self.problem, (i, pi, si), (j, pj, sj), swap_pool=True
                    )

    def __enum_local_move(self: Solution) -> Iterable[LocalMove]:
        # Simple Movements

        # Add (unused) Servers
        for add in self.unused:
            for r in range(self.problem.r):
                for segment in self.problem.rows[r]:
                    if self.__fits(add, segment):
                        for p in range(self.problem.p):
                            yield LocalMove(self.problem, (add, p, segment), None)

        # Remove (used) Servers
        for remove in self.used:
            pool, segment = self.alloc[remove]
            yield LocalMove(self.problem, None, (remove, pool, segment))

        # Change (used) Server Segment
        for server in self.used:
            pool, segment = self.alloc[server]
            for r in range(self.problem.r):
                for seg in self.problem.rows[r]:
                    if seg != segment and self.__fits(server, seg):
                        yield LocalMove(
                            self.problem, (server, pool,
                                           seg), (server, pool, segment)
                        )

        # Change (used) Server Pool
        for server in self.used:
            pool, segment = self.alloc[server]
            for p in range(self.problem.p):
                if p != pool:
                    yield LocalMove(
                        self.problem, (server, p,
                                       segment), (server, pool, segment)
                    )

        # Complex Movements

        # Swap (used) servers pools/segments
        for i in self.used:
            for j in self.used:
                if i != j:
                    pi, si = self.alloc[i]
                    pj, sj = self.alloc[j]
                    if si != sj and self.__swappable(i, j):
                        yield LocalMove(
                            self.problem, (i, pi, si), (j, pj,
                                                        sj), swap_segment=True
                        )
                    if pi != pj:
                        yield LocalMove(
                            self.problem, (i, pi, si), (j, pj,
                                                        sj), swap_pool=True
                        )

    def __init_bound(self: Solution,) -> tuple[float, list[int], list[int], list[int], list[float]]:
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

    def __upper_bound_update_add(
        self: Solution,
        c: Component,
        ub_kp: list[int],
        ub_lim: list[int],
        ub_full: list[int],
        ub_frac: list[int],
    ) -> float:
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

    def __upper_bound_update_remove(
        self: Solution,
        c: Component,
        ub_kp: list[int],
        ub_lim: list[int],
        ub_full: list[int],
        ub_frac: list[int],
    ) -> float:
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

    def __row_upper_bound(self: Solution, row: int, total: float, ub: float):
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

    def __non_repeating_lcg(self: Solution, n: int) -> Iterable[int]:
        if n > 0:
            a = 5
            m = 1 << math.ceil(math.log2(n))
            if m > 1:
                c = random.randrange(1, m, 2)
                x = random.randrange(m)
                for _ in range(m):
                    if x < n:
                        yield x
                    x = (a * x + c) % m
            else:
                yield 0
                            
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

    def __str__(self: Solution) -> str:
        s, slots = "", [
            self.problem.segments[i][1] for i in range(len(self.problem.segments))
        ]
        for server, allocation in enumerate(self.alloc):
            if allocation is not None:
                pool, segment = allocation
                row = self.problem.segments[segment][0]
                s += f"{row} {slots[segment]} {pool}\n"
                slots[segment] += self.problem.servers[server][0]
            else:
                s += "x\n"
        return s[:-1]