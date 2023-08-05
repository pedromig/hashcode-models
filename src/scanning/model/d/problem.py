from __future__ import annotations

from .solution import Solution
from .component import Component


import time
import operator


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
     
    def random_solution(self: Problem) -> Solution:
      solution = self.empty_solution()
      while (c := solution.random_add_move()) is not None:
        solution.add(c)
      return solution
     
    def heuristic_solution(self: Problem) -> Solution: 
      solution = self.empty_solution()
      
      #while len(libraries := sorted(
      #    set(range(self.l)).symmetric_difference(solution.libraries), 
      #    key = lambda lib: sum([self.scores[x] for x in self.sbooks[lib]][:(self.d - solution.day - self.signup[lib]) * self.rate[lib]]) / self.signup[lib],
      #    reverse=True)):
      #  if solution.day + self.signup[libraries[0]] < self.d:
      #    solution.add(Component(self, libraries[0]))
      #  else:
      #    break
      
      # while len(libraries := sorted(set(range(self.l)).symmetric_difference(solution.libraries), key = lambda l: max(0, min(self.rate[l] * len(self.books[l]), self.d) - self.signup[l] - solution.day), reverse=True)):
      #   if solution.day + self.signup[libraries[0]] < self.d:
      #     solution.add(Component(self, libraries[0]))
      #   else:
      #     break
        
      # while len(libraries := sorted(set(range(self.l)).symmetric_difference(solution.libraries), key = lambda l: (self.signup[l], self.b - len(set(self.books[l]))))):
      #   if solution.day + self.signup[libraries[0]] < self.d:
      #     solution.add(Component(self, libraries[0]))
      #   else:
      #     break
       
      # used = set()
      # while len(libraries := sorted(
      #     set(range(self.l)).symmetric_difference(solution.libraries), 
      #     key = lambda l: sum(
      #       [self.scores[x] for x in self.sbooks[l] if x not in used]
      #       [:(self.d - solution.day - self.signup[l]) * self.rate[l]]
      #       ) / self.signup[l],
      #     reverse=True)):
      #   if solution.day + self.signup[libraries[0]] < self.d:
      #     for book in [b for b in self.sbooks[libraries[0]] if b not in used][:(self.d - solution.day - self.signup[libraries[0]]) * self.rate[libraries[0]]]:
      #       used.add(book)
      #     solution.add(Component(self, libraries[0]))
      #   else:
      #     break
      
      # used = set()
      # while len(libraries := sorted(
      #     set(range(self.l)).symmetric_difference(solution.libraries), 
      #     key = lambda l: len(x for x in self.sbooks[l] if x not in used), reverse=True)):
      #   if solution.day + self.signup[libraries[0]] < self.d:
      #     for book in [b for b in self.sbooks[libraries[0]] if b not in used][:(self.d - solution.day - self.signup[libraries[0]]) * self.rate[libraries[0]]]:
      #       used.add(book)
      #     solution.add(Component(self, libraries[0]))
      #   else:
      #     break
      
      while c := next(solution.enum_heuristic_add_move(), None):
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
        sbooks[lib] = tuple(sorted(self.books[lib], key = lambda b: self.scores[b], reverse=True))
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