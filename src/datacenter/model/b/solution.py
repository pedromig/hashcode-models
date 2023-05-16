from __future__ import annotations
from typing import TYPE_CHECKING, Iterable

from .component import *
from .move import *

if TYPE_CHECKING:
    from .problem import *

import math
import random
import operator
import copy
import sys

class Solution:
    def __init__(
        self: Solution,
        problem: Problem,
        used: set[int] | None,
        unused: set[int] | None,
        forbidden: set[int] | None,
        alloc: list[tuple[int]] | None,
        ss: list[int] | None,
        gc: list[int] | None,
        cp: list[int] | None,
        rc: list[list[int]] | None,
        mc: list[int] | None,
        objv: tuple[int] | None,
        ub: tuple[float] | None,
        ub_kp: list[int] | None,
        ub_lim: list[int] | None,
        ub_full: list[int] | None,
        ub_frac: list[float] | None,
        iserver: int | None,
        init_ub: bool = False,
    ) -> None:

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
            self.forbidden.copy(),
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
            self.iserver,
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
        if self.iserver < self.problem.m:
            server = self.problem.sservers[self.iserver]
            for pool in range(self.problem.p):
                for row in range(self.problem.r):
                    for segment in self.problem.rows[row]:
                        if self.__fits(server, segment):
                            yield Component(self.problem, server, pool, segment)
            yield Component(self.problem, server)

    def enum_remove_move(self: Solution) -> Iterable[Component]:
        for server in self.unused:
            pool, segment = self.alloc[server]
            yield Component(problem, server, pool, segment)

    def enum_heuristic_add_move(self: Solution) -> Iterable[Component]:
        if self.iserver < self.problem.m:
            server = self.problem.sservers[self.iserver]
            for pool in sorted(range(self.problem.p), key=lambda x: self.gc[x]):
                for row in sorted(
                    range(self.problem.r),
                    key=lambda x: self.rc[pool][x]
                ):
                    for segment in self.problem.rows[row]:
                        if self.__fits(server, segment):
                            yield Component(self.problem, server, pool, segment)
            yield Component(self.problem, server)

    def enum_local_move_lcg(self: Solution) -> Iterable[LocalMove]:
        used = list(self.used)
        unused = list(self.unused)

        add_moves = len(self.unused) * \
            len(self.problem.segments) * self.problem.p
        remove_moves = len(self.used)
        change_segment_moves = len(self.used) * len(self.problem.segments)
        change_pool_moves = len(self.used) * self.problem.p
        swap_moves = 2 * (len(used)**2)

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
                        self.problem,
                        (server, pool, seg),
                        (server, pool, segment)
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
                        self.problem,
                        (server, p, segment),
                        (server, pool, segment)
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
                        self.problem,
                        (i, pi, si),
                        (j, pj, sj),
                        swap_segment=True
                    )
            else:
                if pi != pj:
                    yield LocalMove(
                        self.problem,
                        (i, pi, si),
                        (j, pj, sj),
                        swap_pool=True
                    )

    def enum_local_move(self: Solution) -> Iterable[LocalMove]:
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
                            self.problem,
                            (server, pool, seg),
                            (server, pool, segment)
                        )

        # Change (used) Server Pool
        for server in self.used:
            pool, segment = self.alloc[server]
            for p in range(self.problem.p):
                if p != pool:
                    yield LocalMove(
                        self.problem,
                        (server, p, segment),
                        (server, pool, segment)
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
                            self.problem,
                            (i, pi, si),
                            (j, pj, sj),
                            swap_segment=True
                        )
                    if pi != pj:
                        yield LocalMove(
                            self.problem,
                            (i, pi, si),
                            (j, pj, sj),
                            swap_pool=True
                        )

    def enum_random_local_move_wor(self: Solution) -> Iterable[LocalMove]:
        moves = list(self.enum_local_move())
        random.shuffle(moves)
        for move in moves:
            yield move

    def enum_random_local_move_wor_lcg(self: Solution) -> Iterable[LocalMove]:
        moves = list(self.enum_local_move_lcg())
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

    def random_local_move_lcg(self: Solution) -> LocalMove:
        moves = list(self.enum_local_move_lcg())
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

        # objv = self._objective_value(move)

        # Update Objective Value
        self.objv = tuple(sorted(self.gc))

        # assert self.objv == objv

        # Update Bound
        self.ub = None

    def perturb(self: Solution, kick) -> None:
        for _ in range(kick):
            self.step(self.random_local_move())

    def add(self: Solution, c: Component) -> None:
        if c.pool is None or c.segment is None:
            # Test
            # ub = self._upper_bound_forbid(c)

            # Forbid Component
            self.__forbid(c)

            # Update Bound
            self.ub = self.__upper_bound_update_forbid(
                c, self.ub_kp, self.ub_lim, self.ub_full, self.ub_frac
            )
        else:
            # Test
            # ub = self._upper_bound_add(c)

            # Add Component
            self.__add(c)

            # Update Objective Value
            self.objv = tuple(sorted(self.gc))

            # Update Bound
            self.ub = self.__upper_bound_update_add(
                c, self.ub_kp, self.ub_lim, self.ub_full, self.ub_frac
            )

        # Test
        # assert ub == self.ub
        # assert self.objv == self._objective_value()

        # Use Next Server
        self.iserver += 1

    def remove(self: Solution, c: Component) -> None:
        # Test
        ub = self._upper_bound_remove(c)

        # Remove Component
        self.__remove(c)

        # Update Objective Value
        self.objv = tuple(sorted(self.gc))

        # Update Bound
        self.ub = self.__upper_bound_update_remove(
            c, self.ub_kp, self.ub_lim, self.ub_full, self.ub_frac
        )
        # assert ub == self.ub

    def objective_increment_add(self: Solution, c: Component) -> tuple[int]:
        cp = self.cp[c.pool] + self.problem.servers[c.server][1]
        rc = (
            self.rc[c.pool][self.problem.segments[c.segment][0]]
            + self.problem.servers[c.server][1]
        )
        gc = self.gc.copy()
        gc[c.pool] = cp - max(self.mc[c.pool], rc)
        gc.sort()
        return tuple(map(operator.sub, tuple(gc), self.objv))

    def objective_increment_remove(self: Solution, c: Component) -> tuple[int]:
        ...

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
        if c.pool is None or c.segment is None:
            ub = self.__upper_bound_update_forbid(
                c,
                self.ub_kp.copy(),
                self.ub_lim.copy(),
                self.ub_full.copy(),
                self.ub_frac.copy(),
            )
        else:
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
        return tuple(map(operator.sub, ub, self.ub))

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

        return tuple(map(operator.sub, ub, self.ub))

    def _objective_value(self: Solution):
        cp = [0] * self.problem.p
        rc = [[0] * len(self.problem.rows) for _ in range(self.problem.p)]
        for i, alloc in enumerate(self.alloc):
            if alloc is not None:
                pool, segment = alloc
                _, capacity = self.problem.servers[i]
                row, _, _ = self.problem.segments[segment]
                cp[pool] += capacity
                rc[pool][row] += capacity
        return tuple(sorted(cp[p] - max(rc[p]) for p in range(self.problem.p)))

    def _score(self: Solution):
        return self._objective_value()[0]

    def _upper_bound_add(self: Solution, c: Component):
        size, capacity = self.problem.servers[c.server]
        row, _, _ = self.problem.segments[c.segment]

        self.cp[c.pool] += capacity
        self.rc[c.pool][row] += capacity

        # Available space on each knapsack
        knapsacks = [0] * self.problem.r
        for i in range(len(self.ss)):
            r, _, sz = self.problem.segments[i]
            for j in range(self.problem.r):
                knapsacks[j] += sz if j != r else 0

        # Knapsack allocated capacity
        capacities = [0] * self.problem.r
        for i in self.used:
            sz, cap = self.problem.servers[i]
            r, _, _ = self.problem.segments[self.alloc[i][1]]
            for j in range(self.problem.r):
                capacities[j] += cap if j != r else 0
                knapsacks[j] -= sz if j != r else 0

        for i in range(self.problem.r):
            capacities[i] += capacity * \
                min(1, (knapsacks[i] / size)) if i != row else 0
            knapsacks[i] -= size if i != row else 0

        # Upper Bound
        ub = [sys.maxsize] * self.problem.p
        for i in range(self.problem.r):
            total = capacities[i]
            for s in self.problem.sservers:
                if s in self.unused and s != c.server and s not in self.forbidden:
                    sz, cp = self.problem.servers[s]
                    if sz <= knapsacks[i]:
                        total += cp
                        knapsacks[i] -= sz
                    else:
                        total += cp * (knapsacks[i] / sz)
                        break
            self.__row_upper_bound(i, total, ub)

        # Decrement Capacities (for bound calculation)
        self.cp[c.pool] -= capacity
        self.rc[c.pool][row] -= capacity

        return tuple(sorted(ub))

    def _upper_bound_remove(self: Solution, c: Component):
        size, capacity = self.problem.servers[c.server]
        row, _, _ = self.problem.segments[c.segment]

        # Decrement Capacity (For bound calculation)
        self.cp[c.pool] -= capacity
        self.rc[c.pool][row] -= capacity

        # Available space on each knapsack
        knapsacks = [0] * self.problem.r
        for i in range(len(self.ss)):
            r, _, sz = self.problem.segments[i]
            for j in range(self.problem.r):
                knapsacks[j] += sz if j != r else 0

        # Knapsack allocated capacity
        capacities = [0] * self.problem.r
        for i in self.used:
            sz, cap = self.problem.servers[i]
            r, _, _ = self.problem.segments[self.alloc[i][1]]
            if i == c.server:
                continue
            for j in range(self.problem.r):
                capacities[j] -= cap if j != r else 0
                knapsacks[j] += sz if j != r else 0

        # Upper Bound
        ub = [sys.maxsize] * self.problem.p
        for i in range(self.problem.r):
            total = capacities[i]
            for s in self.problem.sservers:
                if s == c.server or (s in self.unused and s not in self.forbidden):
                    sz, cp = self.problem.servers[s]
                    if sz <= knapsacks[i]:
                        total += cp
                        knapsacks[i] -= sz
                    else:
                        total += cp * (knapsacks[i] / sz)
                        break
            self.__row_upper_bound(i, total, ub)

        # Increment Capacity (For bound calculation)
        self.cp[c.pool] += capacity
        self.rc[c.pool][row] += capacity

        return tuple(sorted(ub))

    def _upper_bound_forbid(self: Solution, c: Component):
        # Available space on each knapsack
        knapsacks = [0] * self.problem.r
        for i in range(len(self.ss)):
            r, _, sz = self.problem.segments[i]
            for j in range(self.problem.r):
                knapsacks[j] += sz if j != r else 0

        # Knapsack allocated capacity
        capacities = [0] * self.problem.r
        for i in self.used:
            sz, cap = self.problem.servers[i]
            r, _, _ = self.problem.segments[self.alloc[i][1]]
            for j in range(self.problem.r):
                capacities[j] += cap if j != r else 0
                knapsacks[j] -= sz if j != r else 0

        # Upper Bound
        ub = [sys.maxsize] * self.problem.p
        for i in range(self.problem.r):
            total = capacities[i]
            for s in self.problem.sservers:
                if s in self.unused and s != c.server and s not in self.forbidden:
                    sz, cp = self.problem.servers[s]
                    if sz <= knapsacks[i]:
                        total += cp
                        knapsacks[i] -= sz
                    else:
                        total += cp * (knapsacks[i] / sz)
                        break
            self.__row_upper_bound(i, total, ub)

        return tuple(sorted(ub))

    def __add(self: Solution, c: Component):
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
        self.cp[c.pool] -= capacity
        self.rc[c.pool][row] -= capacity
        self.mc[c.pool] = max(self.rc[c.pool])
        self.gc[c.pool] = self.cp[c.pool] - self.mc[c.pool]

    def __forbid(self: Solution, c: Component):
        # Forbid Server
        self.unused.remove(c.server)
        self.forbidden.add(c.server)

    def __fits(self: Solution, server: int, segment: int) -> bool:
        return self.problem.servers[server][0] <= self.ss[segment]

    def __swappable(self: Solution, i: int, j: int) -> bool:
        return (
            self.ss[self.alloc[i][1]] - self.problem.servers[i][0]
            >= self.problem.servers[j][0]
            and self.ss[self.alloc[j][1]] - self.problem.servers[j][0]
            >= self.problem.servers[i][0]
        )

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

    def __init_bound(self: Solution) -> tuple[tuple[float], list[int], list[int], list[int], list[float]]:
        # Knapsacks
        knapsacks = [0] * self.problem.r
        for r, row in enumerate(self.problem.rows):
            for i in row:
                for j in range(self.problem.r):
                    knapsacks[j] += self.problem.segments[i][2] if r != j else 0

        # Bound Initialization
        ub = [sys.maxsize] * self.problem.p
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
            aux = (full[i] + frac[i]) / self.problem.p
            for j in range(self.problem.p):
                ub[j] = min(ub[j], aux)

        return tuple(sorted(ub)), kp, lim, full, frac

    def __upper_bound_update_add(
        self: Solution,
        c: Component,
        ub_kp: list[int],
        ub_lim: list[int],
        ub_full: list[int],
        ub_frac: list[int],
    ) -> float:
        ub = [sys.maxsize] * self.problem.p
        size, capacity = self.problem.servers[c.server]
        row, _, _ = self.problem.segments[c.segment]

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
                    if sx in self.unused and sx not in self.forbidden:
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
                    if (
                        sx in self.unused
                        and sx != c.server
                        and sx not in self.forbidden
                    ):
                        if sz <= ub_kp[i]:
                            ub_kp[i] -= sz
                            ub_full[i] += cap
                        else:
                            ub_frac[i] = cap * (ub_kp[i] / sz)
                            break
                    ub_lim[i] += 1
            total = ub_full[i] + ub_frac[i]
            self.__row_upper_bound(i, total, ub)
        return tuple(sorted(ub))

    def __upper_bound_update_remove(
        self: Solution,
        c: Component,
        ub_kp: list[int],
        ub_lim: list[int],
        ub_full: list[int],
        ub_frac: list[int],
    ) -> float:
        ub = [sys.maxsize] * self.problem.p
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
                    if sx in self.unused and sx not in self.forbidden:
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
                    if sx in self.unused and sx not in self.forbidden:
                        ub_full[i] -= cap
                        ub_kp[i] += sz
                ub_frac[i] = cap * (ub_kp[i] / sz)

            total = ub_full[i] + ub_frac[i]
            self.__row_upper_bound(i, total, ub)
        return tuple(sorted(ub))

    def __upper_bound_update_forbid(
        self: Solution,
        c: Component,
        ub_kp: list[int],
        ub_lim: list[int],
        ub_full: list[int],
        ub_frac: list[int],
    ) -> float:
        ub = [sys.maxsize] * self.problem.p
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
                    if (
                        sx in self.unused
                        and sx != c.server
                        and sx not in self.forbidden
                    ):
                        if sz <= ub_kp[i]:
                            ub_kp[i] -= sz
                            ub_full[i] += cap
                        else:
                            ub_frac[i] = cap * (ub_kp[i] / sz)
                            break
                    ub_lim[i] += 1
            total = ub_full[i] + ub_frac[i]
            self.__row_upper_bound(i, total, ub)
        return tuple(sorted(ub))

    def __row_upper_bound(self: Solution, row: int, total: float, ub: list[float]):
        count = self.problem.p
        nignored = list(range(self.problem.p))
        updates = 1
        while updates > 0:
            updates = 0
            i = 0
            while i < len(nignored):
                p = nignored[i]
                cap = self.cp[p] - self.rc[p][row]
                if cap > total / count + 1e-8:
                    ub[p] = min(ub[p], cap)
                    count -= 1
                    total -= cap
                    updates += 1
                    nignored[i] = nignored[-1]
                    nignored.pop()
                else:
                    i += 1
        aux = total / count
        for i in nignored:
            ub[i] = min(ub[i], aux)

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
