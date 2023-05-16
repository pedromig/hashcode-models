from __future__ import annotations

from .solution import Solution


class Problem:
    def __init__(self: Problem, r: int, s: int, u: int, p: int, m: int,
                 unavailable: tuple[tuple], servers: tuple[tuple]) -> None:
        # Instance Parameters
        self.r, self.s, self.u, self.p, self.m = r, s, u, p, m
        self.unavailable = unavailable
        self.servers = servers

        # Setup
        self.__init_problem()

    def empty_solution(self: Problem) -> Solution:
        return Solution(self, init_ub=True)

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
        return s

    def __str__(self: Problem) -> str:
        s = f"{self.r} {self.s} {self.u} {self.p} {self.m}\n"
        s += "\n".join(f"{row} {slot}" for row,
                       slot in self.unavailable) + "\n"
        s += "\n".join(f"{size} {capacity}" for size, capacity in self.servers)
        return s
