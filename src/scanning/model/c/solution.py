from __future__ import annotations
from typing import TYPE_CHECKING, Generator, Generator

from .component import Component

from .component import *
from .move import *

if TYPE_CHECKING:
    from .problem import *
    
import time
import copy
import random
import math

# TODO: Bound still incorrect

class Solution:
    def __init__(self: Solution, problem: Problem, books: list[set[int]] | None = None,
        quota: list[int] | None = None, used: dict[int] | None = None, 
        ignore: set[int] | None = None,
        forbidden: set[int] | None = None, count: list[int] | None = None,
        libraries: set[int] | None = None, open: int | None = None,
        closed: set[int] | None = None, day: int | None = None, 
        start: list[int] | None = None, objv: int | None = None,
        ub: float | None = None, ub_kp: int | None = None, 
        ub_lim: int | None = None, ub_full: int | None = None, 
        ub_rate: int | None = None,
        init_ub: bool = False) -> None:
      
      # Problem
      self.problem = problem

      # Solution

      ## Quota
      self.quota = quota if quota is not None else [0] * self.problem.l # Available Quota / Library
        
      ## Signup 
      self.day = day if day is not None else 0 # Last signup
      self.start = start if start is not None else [self.problem.d] * self.problem.l # Start Day / library

      ## Libraries
      self.libraries = libraries if libraries is not None else set() # Signed Libraries    
      self.ignore = ignore if ignore is not None else set() # Ignored Libraries
      self.open = open if open is not None else None # Current (Open) Library
      self.closed = closed if closed is not None else set() # Closed Libraries
    
      ## Assignment
      self.books = books if books is not None else [set() for _ in range(self.problem.l)] # Books / Library 
      self.count = count if count is not None else list(map(len, self.problem.libraries))
      
      self.used = used if used is not None else dict() # Used Books 
      self.forbidden = forbidden if forbidden is not None else set() # Forbidden Books 

      # Objective Value
      self.objv = objv if objv is not None else 0 # Objective Value
    
      # Upper Bound
      if ub is None and init_ub: 
        self.ub, self.ub_kp, self.ub_lim, self.ub_full, self.ub_rate = self.__init_upper_bound()
      else:
        self.ub, self.ub_kp, self.ub_lim, self.ub_full, self.ub_rate = ub, ub_kp, ub_lim, ub_full, ub_rate
          
    def copy(self: Solution):
      return Solution(self.problem, copy.deepcopy(self.books), self.quota.copy(),
        copy.deepcopy(self.used), self.libraries.copy(),
        self.day, self.start.copy(), self.objv, self.ub, self.ub_kp,
        self.ub_lim, self.ub_full, False)

    def feasible(self: Solution) -> bool:
      return True

    def score(self: Solution) -> int:
      return self.objv 

    def objective_value(self: Solution) -> int:
      return self.objv

    def upper_bound(self: Solution) -> int:
      return self.ub

    def enum_add_move(self: Solution) -> Generator[Solution]:
      if self.__signed(self.open) and self.__has_quota(self.open):
        for book in self.problem.books[self.open]:
          if book not in self.books[self.open] and book not in self.used:
            yield Component(self.problem, book, self.open)
 
      for l in range(self.problem.l):
        if l not in self.closed and l != self.open and self.__signable(l):
          yield Component(self.problem, None, l)
  
    def enum_heuristic_add_move(self: Solution) -> Generator[Solution]:      
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
              
    def enum_remove_move(self: Solution) -> Generator[Solution]:
      for book, library in self.used.items():
        if self.__signed(library):
          for book in self.books[library]:
            yield Component(self.problem, book, library)
          yield Component(self.problem, None, library)
          
    def enum_local_move_lcg(self: Solution) -> Generator[LocalMove]:
      libraries = list(self.libraries)
      used = list(self.used.keys())
      
      add_book_moves = len(libraries) * self.problem.b
      remove_book_moves = self.problem.b
      swap_book_moves = len(libraries) * (self.problem.b**2)
      add_swap_book_moves = len(self.used) * self.problem.l * (self.problem.b + 1)
      
      for move in self.__non_repeating_lcg(add_book_moves + remove_book_moves + swap_book_moves + add_swap_book_moves):
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
      
    def enum_local_move(self: Solution) -> Generator[LocalMove]:
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
              
    def enum_random_local_move_wor(self: Solution) -> Generator[LocalMove]:
      def default(self: Solution) -> Generator[LocalMove]:
        moves = list(self.enum_local_move())
        random.shuffle(moves)
        for move in moves:
          yield move
          
      def lcg(self: Solution) -> Generator[LocalMove]:
        for move in self.enum_local_move_lcg():
          yield move

      return lcg(self)
 
    def random_add_move(self: Solution) -> Component:
      libraries = list(range(self.problem.l))
      random.shuffle(libraries)
      for library in libraries:
        if self.__signed(library) and self.__has_quota(library):
          books = list(self.problem.books[library])
          random.shuffle(books)
          for book in books:
            if book not in self.books[library] and book not in self.used:
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
      def default(self: Solution) -> LocalMove:
        moves = list(self.enum_local_move())
        random.shuffle(moves)
        for move in moves:
          return move
        
      def lcg(self: Solution) -> LocalMove:
        return next(self.enum_local_move_lcg()) 
      
      return lcg(self)
      
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
        self
      # Update Objective Value
      self.objv = objv
     
      # Update Bound
      self.ub = None
 
    def perturb(self: Solution, kick: int) -> None:
      for _ in range(kick):
        self.step(self.random_local_move())

    def add(self: Solution, c: Component) -> None: 
      # Aux
      ub_library = self.open 
      ub_day = self.day if c.book is None else self.day - self.problem.signup[c.library]
      
      # Add Component
      score = self.__add(c)

      # Update Objective Value
      self.objv += score 

      # Update Bound
      self.ub, self.ub_kp, self.ub_lim, self.ub_full, self.ub_rate = self.__upper_bound_update_add(c, ub_day, ub_library, self.ub_kp, self.ub_lim, self.ub_full, self.ub_rate, self.count, self.forbidden, self.ignore)
      
      # print(c, self.ub, self.ub_kp, self.ub_lim, self.ub_full, self.ub_rate)
             
    def remove(self: Solution, c: Component) -> None:
      # Remove Component
      score = self.__remove(c)
      
      # Update Objective Value
      self.objv -= score         
        
      # Update Bound
      self.ub, self.ub_kp, self.ub_lim, self.ub_full, self.ub_rate = self.__upper_bound_update_remove(c, self.ub_kp, self.ub_lim, self.ub_full, self.ub_rate)
      
    def objective_increment_add(self: Solution, c: Component) -> int:  
      objv = self.objv
      if not self.__is_library(c) and c.book not in self.used:
          objv += self.problem.scores[c.book]
      return objv - self.objv

    def objective_increment_remove(self: Solution, c: Component) -> int:
      objv = self.objv
      if not self.__is_library(c) and c.book in self.used:
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
      ub = self.__upper_bound_update_add(c, self.day, self.open, self.ub_kp, self.ub_lim, self.ub_full, self.ub_rate, self.count.copy(), self.forbidden.copy(), self.ignore.copy())[0]
      return ub - self.ub
   
    def upper_bound_increment_remove(self: Solution, c: Component) -> int:
      ub = self.__upper_bound_update_remove(c, self.day, self.open, self.ub_kp, self.ub_lim, self.ub_full, self.ub_rate)[0]
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
      
      if self.__is_library(c): 
        # Add Library
        self.libraries.add(c.library)
        
        # Close Last Library 
        if self.open is not None: 
          self.closed.add(self.open)
        self.open = c.library
         
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
      
      if self.__is_library(c): 
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
     
    def __is_library(self: Solution, c: Component) -> bool:
      return c.book is None
    
    def __init_upper_bound(self: Solution) -> tuple[int, int, int, int]:
      signable = list(filter(lambda x: self.__signable(x), range(self.problem.l)))
      cap = sum(self.problem.rate[l] * self.problem.d for l in signable)
      full = sum(list(map(lambda b: self.problem.scores[b], self.problem.sbooks))[:cap])
      lim = min(self.problem.b, cap)
      kp = cap - lim
      rate = sum(self.problem.rate[i] for i in signable)
      return full, kp, lim, full, rate
     
    def __close_library(self: Solution, day: int, library: int,
                        ub_kp: int, ub_lim: int, ub_full: int,
                        count: list[int], forbidden: set[int]) -> tuple[int, int]:
      for book in self.problem.books[library]:
        if book in self.used:
          continue
        count[book] -= 1
        if count[book] == 0:
          forbidden.add(book)
          if self.problem.pbooks[book] < ub_lim:
            ub_kp += 1
            ub_full -= self.problem.scores[book]
            while ub_lim < self.problem.b:
              b = self.problem.sbooks[ub_lim]
              if b not in self.used and b not in forbidden:
                if ub_kp >= 1:
                  ub_kp -= 1
                  ub_full += self.problem.scores[b] 
                else:
                  break
              ub_lim += 1
      ub_kp -= ((self.problem.rate[library] * (self.problem.d - day)) - len(self.books[library])) 
      return ub_kp, ub_full, ub_lim
      
    def __upper_bound_update_add(self: Solution, c: Component, day: int, library: int,
                                 ub_kp: int, ub_lim: int, ub_full: int, ub_rate: int,
                                 count: list[int], forbidden: set[int], ignore: set[int]) -> int:
      if self.__is_library(c): 
        if library is not None:
          ub_kp, ub_full, ub_lim = self.__close_library(day, library, ub_kp, ub_lim, ub_full, count, forbidden)
          
        for l in range(self.problem.l):
          if l not in self.libraries and l not in ignore and (day + self.problem.signup[c.library]) >= self.problem.d:
            ignore.add(l)
            ub_kp -= self.problem.rate[l] * self.problem.d
            ub_rate -= self.problem.rate[l]
          
        ub_kp -= min(self.problem.signup[c.library], self.problem.d - day) * ub_rate 
        ub_rate -= self.problem.rate[c.library]
        while ub_kp < 0 and ub_lim > 0:
          ub_lim -= 1
          book = self.problem.sbooks[ub_lim]
          if book not in self.used and book not in forbidden:
            ub_kp += 1
            ub_full -= self.problem.scores[book] 
      else:
        if self.problem.pbooks[c.book] >= ub_lim:
          ub_kp -= 1
          ub_full += self.problem.scores[c.book]
          while ub_kp < 0 and ub_lim > 0:
            ub_lim -= 1
            book = self.problem.sbooks[ub_lim]
            if book not in self.used and book not in forbidden:
              ub_kp += 1
              ub_full -= self.problem.scores[book]
      return ub_full, ub_kp, ub_lim, ub_full, ub_rate
    
    def __upper_bound_update_remove(self: Solution, c: Component, 
                                    ub_kp: int, ub_lim: int, ub_full: int, 
                                    ub_rate: int, ub_count: list[int], ub_forbidden: set[int]) -> int:
      ...
    
    def __non_repeating_lcg(self: Solution, n: int) -> Generator[int]:
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
      s = f"quota = {self.quota}\n"
      s += f"day = {self.day}\n" 
      s += f"start = {self.start}\n"
      s += f"libraries = {self.libraries}\n"
      s += f"books = {self.books}\n"
      s += f"used = {self.used}\n"
      s += f"objv = {self.objv}\n"
      s += f"ub = {self.ub}\n"
      s += f"ub_kp = {self.ub_kp}\n"
      s += f"ub_lim = {self.ub_lim}\n"
      s += f"ub_full = {self.ub_full}\n"
      return s
    
    def __str__(self: Solution) -> str: 
      s = f"{len(self.libraries)}\n"
      for library in self.libraries:
        s += f"{library} {len(self.books[library])}\n"
        s += f"{' '.join(str(book) for book in self.books[library])}\n"
      return s[:-1]