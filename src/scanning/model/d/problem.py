from __future__ import annotations

from .solution import Solution
from .component import Component
import time

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
      return Solution(self)
    
    def heuristic_solution(self: Problem) -> Solution: 
      solution = self.empty_solution()
      
      while len(libraries := sorted(
          set(range(self.l)).symmetric_difference(solution.libraries), 
          key = lambda lib: sum([self.scores[x] for x in self.sbooks[lib]][:(self.d - solution.day - self.signup[lib]) * self.rate[lib]]) / self.signup[lib],
          reverse=True)):
        if solution.day + self.signup[libraries[0]] < self.d:
          solution.add(Component(self, None, libraries[0]))
        else:
          break
      
      solution.objv , solution.used, solution.books, solution.quota, solution.freq = solution.assign_books_optimally(solution.freq, init=True)
      
      # for book in sorted(range(self.b), key = lambda b: self.scores[b] / len(self.libraries[b]) if len(self.libraries[b]) != 0 else 0, reverse=False):
      #   for library in self.libraries[book]:
      #     if library in solution.libraries and solution.quota[library] > 0 and book not in solution.used and book not in solution.books[library]:
      #       solution.add(Component(self, book, library))
      
      # for library in solution.libraries: 
      #   for book in self.sbooks[library]: 
      #     if solution.quota[library] > 0 and book not in solution.used: 
      #       solution.add(Component(self, book, library))
       
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
      assert len(scores) == b, False
      
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
        assert len(set(self.books[lib])) == len(self.books[lib])
        sbooks[lib] = tuple(sorted(self.books[lib], 
                                   key = lambda b: (self.scores[b], self.l - len(libraries[b])), 
                                   reverse=False))
        pbooks[lib] = dict()
        for i in range(len(sbooks[lib])):
          pbooks[lib][sbooks[lib][i]] = i 
      
      self.libraries = tuple(libraries)  
      self.sbooks = tuple(sbooks) 
      self.pbooks = tuple(pbooks)
      
    def __repr__(self: Problem) -> None:
      s = f"b = {self.b}\n"
      s += f"l = {self.l}\n"
      s += f"d = {self.d}\n"
      s += f"scores = {self.scores}\n"
      s += f"books = {self.books}\n"
      s += f"libraries = {self.libraries}\n" 
      s += f"size = {self.size}\n"
      s += f"signup = {self.signup}\n"
      s += f"rate = {self.rate}\n"
      s += f"sbooks = {self.sbooks}\n"
      s += f"pbooks = {self.pbooks}\n"
      return s

    def __str__(self: Problem) -> str:
        s = f"{self.b} {self.l} {self.d}\n"
        s += " ".join(str(i) for i in self.books) + "\n"
        for i in range(self.l):
          s += f"{self.size[i]} {self.signup[i]} {self.rate[i]}\n" 
          s += " ".join(str(i) for i in self.books[i]) + "\n"
        return s[:-1]