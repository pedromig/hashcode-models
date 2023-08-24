from __future__ import annotations

from typing import Optional, Iterable
from dataclasses import dataclass

import random
import copy
import math

class Problem:
    def __init__(self: Problem, b: int, l: int, d: int, scores: tuple[tuple[int, ...]],
                 books: tuple[tuple[int, ...]], signup: tuple[int, ...],
                 size: tuple[int, ...], rate: tuple[int, ...]) -> None:

        # Instance Parameters
        self.b, self.l, self.d = b, l, d
        self.scores = scores
        self.books = books
        self.size = size
        self.signup = signup
        self.rate = rate

        # Setup
        self.__init_problem()

    def empty_solution(self: Problem) -> Solution:
        return Solution(self, init_ub=True)
    
    def heuristic_solution(self: Problem) -> Solution:
        solution = self.empty_solution()
        while c := next(solution.heuristic_add_moves(), None):
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
        b, l, d = map(int, input().split())

        # Book Scores
        scores = tuple(map(int, input().split()))

        # Load Libraries
        books, size, signup, rate = [None] * l, [0] * l, [0] * l, [0] * l
        for i in range(l):
            size[i], signup[i], rate[i] = map(int, input().split())
            books[i] = tuple(map(int, input().split()))

        return Problem(b, l, d, scores, tuple(books), tuple(signup), tuple(size), tuple(rate))

    def __init_problem(self: Problem) -> None:
        # Books / Library
        libraries = [None] * self.b
        for book in range(self.b):
            b = []
            for library in range(self.l):
                if book in self.books[library]:
                    b.append(library)
            libraries[book] = tuple(b)

        # Sorted Books / Library
        sbooks = [None] * self.l
        pbooks = [None] * self.l
        for lib in range(self.l):
            sbooks[lib] = tuple(sorted(self.books[lib],
                                    key = lambda b: self.scores[b],
                                    reverse=True))
            pbooks[lib] = dict()
            for i in range(len(sbooks[lib])):
                pbooks[lib][sbooks[lib][i]] = i

        self.libraries = tuple(libraries)
        self.sbooks = tuple(sbooks)
        self.pbooks = tuple(pbooks)

    def __str__(self: Problem) -> str:
        s = f"{self.b} {self.l} {self.d}\n"
        s += " ".join(str(i) for i in self.books) + "\n"
        for i in range(self.l):
            s += f"{self.size[i]} {self.signup[i]} {self.rate[i]}\n"
            s += " ".join(str(i) for i in self.books[i]) + "\n"
        return s[:-1]

class Solution:
    def __init__(self: Solution, problem: Problem,
                books: Optional[list[set[int]]] = None, quota: Optional[list[int]] = None,
                used: Optional[dict[int]] = None, libraries: Optional[set[int]] = None,
                day: Optional[int] = None, start: Optional[list[int]] = None,
                objv: Optional[int] = None, ub: Optional[float] = None,
                ub_kp: Optional[list[int]] = None, ub_lim: Optional[list[int]] = None,
                ub_full: Optional[list[int]] = None, init_ub: Optional[bool] = False) -> None:

        # Problem
        self.problem = problem

        # Solution

        ## Quota
        self.quota = quota if quota is not None else [0] * self.problem.l

        # Signup
        self.day = day if day is not None else 0 # Last signup
        self.start = start if start is not None else [self.problem.d] * self.problem.l

        # Libraries
        self.libraries = libraries if libraries is not None else set()

        # Assignment
        self.books = books if books is not None else [set() for _ in range(self.problem.l)]
        self.used = used if used is not None else dict()

        # Objective Value
        self.objv = objv if objv is not None else 0

        # Upper Bound
        if ub is None and init_ub:
            self.ub, self.ub_kp, self.ub_lim, self.ub_full = self.__init_upper_bound()
        else:
            self.ub, self.ub_kp, self.ub_lim, self.ub_full = ub, ub_kp, ub_lim, ub_full

    def copy(self: Solution) -> Solution:
        return Solution(self.problem, copy.deepcopy(self.books), self.quota.copy(),
                        copy.deepcopy(self.used), self.libraries.copy(), self.day,
                        self.start.copy(), self.objv, self.ub, self.ub_kp.copy(),
                        self.ub_lim.copy(), self.ub_full.copy())

    def feasible(self: Solution) -> bool:
        return True

    def score(self: Solution) -> int:
        return self.objv

    def objective_value(self: Solution) -> int:
        return self.objv

    def upper_bound(self: Solution) -> int:
        return self.ub

    def add_moves(self: Solution) -> Iterable[Component]:
        for library in range(self.problem.l):
            if self.__signed(library) and self.__has_quota(library):
                for book in self.problem.books[library]:
                    if book not in self.books[library] and not self.__forbidden(book, library):
                        yield Component(self.problem, book, library)
                        continue
            if self.__signable(library):
                yield Component(self.problem, None, library)

    def heuristic_add_moves(self: Solution) -> Iterable[Component]:
        libraries = sorted(
            set(range(self.problem.l)).symmetric_difference(self.libraries),
            key = lambda lib: sum([self.problem.scores[x] for x in self.problem.sbooks[lib]][:(self.problem.d - self.day - self.problem.signup[lib]) * self.problem.rate[lib]]) / self.problem.signup[lib],
            reverse=True)

        for library in libraries:
            yield Component(self.problem, None, library)

        for library in self.libraries:
            for book in self.problem.sbooks[library]:
                if self.__has_quota(library) and book not in self.used:
                    yield Component(self.problem, book, library)
                elif not self.__has_quota(library):
                    break

    def remove_moves(self: Solution) -> Iterable[Component]:
        for book, library in self.used.items():
            if self.__signed(library):
                for book in self.books[library]:
                    yield Component(self.problem, book, library)
                yield Component(self.problem, None, library)
          
    def local_moves(self: Solution) -> Iterable[LocalMove]:
        # Add (unused) books
        for library in self.libraries:
            if self.__has_quota(library):
                for book in self.problem.books[library]:
                    if book not in self.used:
                        yield LocalMove(self.problem, add=(book, library))

        # Remove (used) books
        for book, library in self.used.items():
            yield LocalMove(self.problem, remove=(book, library))

        # Swap Moves (Same Library)
        for library in self.libraries:
            for i in self.problem.books[library]:
                if i not in self.used:
                    for j in self.books[library]:
                        yield LocalMove(self.problem, add=(i, library), remove=(j, library))

        # Swap (used) books from one library (i) to another (j).
        # Add other (unused) book to library (i) if possible.
        for book in self.used:
            i = self.used.get(book)
            for j in self.problem.libraries[book]:
                if i != j and self.__signed(j) and self.__has_quota(j):
                    yield LocalMove(self.problem, remove=(book, i), add=(book, j))
                    for other in self.problem.books[i]:
                        if other not in self.used:
                            yield LocalMove(self.problem, swap=(book, other, i), add=(book, j))
          

    def random_local_moves_wor(self: Solution) -> Iterable[LocalMove]:
        libraries = list(self.libraries)
        used = list(self.used.keys())

        add_book_moves = len(libraries) * self.problem.b
        remove_book_moves = self.problem.b
        swap_book_moves = len(libraries) * (self.problem.b**2)
        add_swap_book_moves = len(self.used) * self.problem.l * (self.problem.b + 1)

        for move in non_repeating_lcg(add_book_moves + remove_book_moves + swap_book_moves + add_swap_book_moves):
            # Add Moves
            if move < add_book_moves:
                library, book = libraries[move // self.problem.b], move % self.problem.b
                if self.__has_quota(library) and book not in self.used and book in self.problem.books[library]:
                    yield LocalMove(self.problem, add=(book, library))
                continue
            move -= add_book_moves

            # Remove Moves
            if move < remove_book_moves:
                book = move
                if book in self.used:
                    library = self.used.get(book)
                    yield LocalMove(self.problem, remove=(book, library))
                continue
            move -= remove_book_moves

            # Swap Moves
            if move < swap_book_moves:
                library, move = libraries[move // (self.problem.b**2)], move % (self.problem.b**2)
                i, j = move // self.problem.b, move % self.problem.b
                if i not in self.used and i in self.problem.books[library] and j in self.books[library]:
                    yield LocalMove(self.problem, add=(i, library), remove=(j, library))
                continue
            move -= swap_book_moves

            # Swap-Add Moves
            if move < add_swap_book_moves:
                book, move = used[move // (self.problem.l * (self.problem.b + 1))], move % (self.problem.l * (self.problem.b + 1))
                i = self.used.get(book)
                j, other = move // (self.problem.b + 1), move % (self.problem.b + 1)
                if i != j and self.__signed(j) and book in self.problem.books[j] and self.__has_quota(j):
                    if other == 1:
                        yield LocalMove(self.problem, remove=(book, i), add=(book, j))
                    elif other not in self.used and other in self.problem.books[i]:
                        yield LocalMove(self.problem, swap=(book, other, i), add=(book, j))

    def random_add_move(self: Solution) -> Component:
        libraries = list(range(self.problem.l))
        random.shuffle(libraries)
        for library in libraries:
            if self.__signed(library) and self.__has_quota(library):
                books = list(self.problem.books[library])
                random.shuffle(books)
                for book in books:
                    if book not in self.books[library] and not self.__forbidden(book, library):
                        return Component(self.problem, book, library)
                continue
            if self.__signable(library):
                return Component(self.problem, None, library)

    def random_remove_move(self: Solution) -> Component:
        libraries = list(range(self.problem.l))
        random.shuffle(libraries)
        for library in libraries:
            if self.__signed(library):
                books = list(self.books[library])
                random.shuffle(books)
                for book in books:
                    return Component(self.problem, book, library)
                return Component(self.problem, None, library)

    def random_local_move(self: Solution) -> LocalMove:
        return next(self.random_local_moves_wor(), None)

    def add(self: Solution, c: Component) -> None:
        # Add Component
        score = self.__add(c)

        # Update Objective Value
        self.objv += score

        # Update Bound
        self.ub = self.__upper_bound_update_add(c, self.ub_kp, self.ub_lim, self.ub_full)

    def remove(self: Solution, c: Component) -> None:
        # Remove Component
        score = self.__remove(c)

        # Update Objective Value
        self.objv -= score

        # Update Bound
        self.ub = self.__upper_bound_update_remove(c, self.ub_kp, self.ub_lim, self.ub_full)
      
    def step(self: Solution, move: LocalMove) -> None:
        objv = self.objv

        if move.swap is not None:
            book, other, library = move.swap
            objv -= self.__remove(Component(self.problem, book, library))
            objv += self.__add(Component(self.problem, other, library))

        if move.remove is not None:
            objv -= self.__remove(Component(self.problem, *move.remove))

        if move.add is not None:
            objv += self.__add(Component(self.problem, *move.add))

        # Update Objective Value
        self.objv = objv

        # Update Bound
        self.ub = None

    def perturb(self: Solution, kick: int) -> None:
        for _ in range(kick):
            self.step(self.random_local_move())

    def objective_increment_add(self: Solution, c: Component) -> int:
        objv = self.objv
        if c.book is not None and c.book not in self.used:
            objv += self.problem.scores[c.book]
        return objv - self.objv

    def objective_increment_remove(self: Solution, c: Component) -> int:
        objv = self.objv
        if c.book is not None and c.book in self.used:
            objv -= self.problem.scores[c.book]
        return objv - self.objv

    def objective_increment_local(self: Solution, m: LocalMove) -> int:
        objv = self.objv
        if m.remove is not None:
            book, _ = m.remove
            objv -= self.problem.scores[book]

        if m.add is not None:
            book, _ = m.add
            objv += self.problem.scores[book]

        if m.swap is not None:
            book, other, _= m.swap
            objv -= self.problem.scores[book]
            objv += self.problem.scores[other]
        return objv - self.objv

    def upper_bound_increment_add(self: Solution, c: Component) -> int:
        ub = self.__upper_bound_update_add(c, self.ub_kp.copy(), self.ub_lim.copy(), self.ub_full.copy())
        return ub - self.ub

    def upper_bound_increment_remove(self: Solution, c: Component) -> int:
        ub = self.__upper_bound_update_remove(c, self.ub_kp.copy(), self.ub_lim.copy(), self.ub_full.copy())
        return ub - self.ub

    def _objective_value(self: Solution) -> int:
        books = set()
        quota = [(self.problem.d - self.start[i]) * self.problem.rate[i] for i in range(self.problem.l)]
        for library in self.libraries:
            for book in self.books[library]:
                if quota[library] > 0:
                    books.add(book)
                    quota[library] -= 1
        return sum(self.problem.scores[i] for i in books)

    def _score(self: Solution) -> int:
        return self._objective_value()

    def __add(self: Solution, c: Component) -> int:
        score = 0
        if c.book is None:
            # Add Library
            self.libraries.add(c.library)

            # Update Library Start Day and Quota
            self.day += self.problem.signup[c.library]
            self.start[c.library] = self.day
            self.quota[c.library] = (self.problem.d - self.day) * self.problem.rate[c.library]
        else:
            # Update Score and Add Book
            self.books[c.library].add(c.book)

            # Score
            if c.book not in self.used:
                self.used.setdefault(c.book, c.library)
                score = self.problem.scores[c.book]

            # Update Library Quota and Day
            self.quota[c.library] -= 1
        return score

    def __remove(self: Solution, c: Component) -> int:
        score = 0
        if c.book is None:
            # Remove Library
            self.libraries.remove(c.library)

            # Update Library Start Day and Quota
            day = self.start[c.library]
            for lib, start in enumerate(self.start):
                if start > self.start[c.library] and start != self.problem.d:
                    self.start[lib] -= self.problem.signup[c.library]
                    day = max(day, self.start[lib])

            self.day = day
            self.start[c.library] = self.problem.d
            self.quota[c.library] = 0
        else:
            # Update Score and Remove Book
            self.books[c.library].remove(c.book)

            # Score
            if c.book in self.used:
                self.used.pop(c.book)
                score = self.problem.scores[c.book]

            # Update Library Quota and Day
            self.quota[c.library] += 1
        return score

    def __signed(self: Solution, library: int) -> bool:
        return library in self.libraries

    def __signable(self: Solution, library: int) -> bool:
        return not self.__signed(library) and (self.day + self.problem.signup[library]) < self.problem.d

    def __has_quota(self: Solution, library: int) -> bool:
        return self.quota[library] > 0

    def __forbidden(self: Solution, book: int, library) -> bool:
        return book in self.used and library != self.used[book]

    def __init_upper_bound(self: Solution) -> tuple[int, list[int], list[int], list[int]]:
        ub = 0
        lim, full, kp = [0] * self.problem.l, [0] * self.problem.l, [0] * self.problem.l
        for library in range(self.problem.l):
            kp[library] = (self.problem.rate[library] * (self.problem.d - self.problem.signup[library]))
            for book in self.problem.sbooks[library]:
                if kp[library] >= 1:
                    kp[library] -= 1
                    full[library] += self.problem.scores[book]
                    lim[library] += 1
                else:
                    break
            ub += full[library]
        return ub, kp, lim, full

    def __upper_bound_update_add(self: Solution, c: Component,
                                 ub_kp: list[int], ub_lim: list[int],
                                 ub_full: list[int]) -> int:
        ub = 0
        if c.book is None:
            d = self.problem.signup[c.library]
            for library in range(self.problem.l):
                if not self.__signed(library) and library != c.library:
                    ub_kp[library] -= d * self.problem.rate[library]
                    while ub_kp[library] < 0 and ub_lim[library] > 0:
                        ub_lim[library] -= 1
                        book = self.problem.sbooks[library][ub_lim[library]]
                        if not self.__forbidden(book, library):
                            ub_kp[library] += 1
                            ub_full[library] -= self.problem.scores[book]
                ub += ub_full[library]
        else:
            for i in range(self.problem.l):
                if i == c.library and self.problem.pbooks[i][c.book] >= ub_lim[i]:
                    ub_kp[i] -= 1
                    ub_full[i] += self.problem.scores[c.book]
                    while ub_kp[i] < 0 and ub_lim[i] > 0:
                        ub_lim[i] -= 1
                        book = self.problem.sbooks[i][ub_lim[i]]
                        if book not in self.books[i] and not self.__forbidden(book, i):
                            ub_kp[i] += 1
                            ub_full[i] -= self.problem.scores[book]
                elif i != c.library and c.book in self.problem.books[i] and self.problem.pbooks[i][c.book] < ub_lim[i]:
                    ub_kp[i] += 1
                    ub_full[i] -= self.problem.scores[c.book]
                    while ub_lim[i] < len(self.problem.books[i]):
                        book = self.problem.sbooks[i][ub_lim[i]]
                        if book not in self.books[i] and not self.__forbidden(book, i):
                            if ub_kp[i] >= 1:
                                ub_kp[i] -= 1
                                ub_full[i] += self.problem.scores[book]
                            else:
                                break
                        ub_lim[i] += 1
                ub += ub_full[i]
        return ub

    def __upper_bound_update_remove(self: Solution, c: Component,
                                    ub_kp: list[int], ub_lim: list[int],
                                    ub_full: list[int]) -> int:
        ub = 0
        if c.book is None:
            d = self.problem.signup[c.library]
            for library in range(self.problem.l):
                if not self.__signed(library) and library != c.library:
                    ub_kp[library] += d * self.problem.rate[library]
                    while ub_kp[library] > 0 and ub_lim[library] < self.problem.l:
                        book = self.problem.sbooks[library][ub_lim[library]]
                        if not self.__forbidden(book, library):
                            ub_kp[library] -= 1
                            ub_full[library] += self.problem.scores[book]
                        ub_lim[library] += 1
                ub += ub_full[library]
        else:
          for i in range(self.problem.l):
            if i == c.library and self.problem.pbooks[i][c.book] >= ub_lim[i]:
                ub_kp[i] += 1
                ub_full[i] -= self.problem.scores[c.book]
                while ub_lim[i] < len(self.problem.books[i]):
                    book = self.problem.sbooks[i][ub_lim[i]]
                    if book not in self.books[i] and not self.__forbidden(book, i):
                        if ub_kp[i] >= 1:
                            ub_kp[i] -= 1
                            ub_full[i] += self.problem.scores[book]
                        else:
                            break
                    ub_lim[i] += 1
            elif i != c.library and c.book in self.problem.books[i] and self.problem.pbooks[i][c.book] < ub_lim[i]:
                ub_kp[i] -= 1
                ub_full[i] += self.problem.scores[c.book]
                while ub_kp[i] < 0 and ub_lim[i] > 0:
                    ub_lim[i] -= 1
                    book = self.problem.sbooks[i][ub_lim[i]]
                    if book not in self.books[i] and not self.__forbidden(book, i):
                        ub_kp[i] += 1
                        ub_full[i] -= self.problem.scores[book]
            ub += ub_full[i]
        return ub

    def __str__(self: Solution) -> str:
        s = f"{len(self.libraries)}\n"
        for library in self.libraries:
            s += f"{library} {len(self.books[library])}\n"
            s += f"{' '.join(str(book) for book in self.books[library])}\n"
        return s[:-1]

@dataclass(order=True)
class LocalMove:
    problem: Problem 
    add: Optional[tuple[int, int]] = None
    remove: Optional[tuple[int, int]] = None
    swap: Optional[tuple[int, int, int]] = None

@dataclass(order=True)
class Component:
    problem: Problem
    book: Optional[int] = None
    library: int
        
def non_repeating_lcg(n: int, seed: Optional[int] = None) -> Iterable[int]:
    if seed is not None:
        random.seed(seed)
    "Pseudorandom sampling without replacement in O(1) space"
    if n > 0:
        a = 5 # always 5
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