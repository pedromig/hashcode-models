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

# import numpy as np
# import scipy.optimize as optimize
import networkx as nx

# TODO: Plot Instances

class Solution:
    def __init__(self: Solution, problem: Problem, books: list[set[int]] | None = None,
        quota: list[int] | None = None, used: dict[int] | None = None, 
        libraries: set[int] | None = None, order: list[int] | None = None, 
        iorder: dict[int] | None = None,
        freq: dict[int, int] | None = None,
        day: int | None = None, 
        start: list[int] | None = None, objv: int | None = None) -> None:
      
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
      self.order = order if order is not None else list() # Library Signup order
      self.iorder = iorder if iorder is not None else dict() 
      
      self.books = books if books is not None else [set() for _ in range(self.problem.l)] # Books / Library 
      self.used = used if used is not None else dict() # Used Books
      self.freq = freq if freq is not None else dict(zip(range(self.problem.b), list(map(len, self.problem.libraries))))
      
      # Objective Value
      self.objv = objv if objv is not None else 0 # Objective Value
      
    def copy(self: Solution): 
      return Solution(self.problem, copy.deepcopy(self.books), self.quota.copy(),
        copy.deepcopy(self.used), self.libraries.copy(), self.order.copy(),
        copy.deepcopy(self.iorder), copy.deepcopy(self.freq),
        self.day, self.start.copy(), self.objv)

    def feasible(self: Solution) -> bool:
      return True

    def score(self: Solution) -> int:
      return self.objv 

    def objective_value(self: Solution) -> int:
      return self.objv
    
    def upper_bound(self: Solution) -> None:
      return None

    def enum_add_move(self: Solution) -> Generator[Solution]:
      for library in range(self.problem.l):
        if self.__signed(library) and self.__has_quota(library):
          for book in self.problem.books[library]:
            if book not in self.books[library] and not self.__forbidden(book, library):
              yield Component(self.problem, book, library)
              continue
        if self.__signable(library):
          yield Component(self.problem, None, library)
  
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
          
    def enum_local_move(self: Solution) -> Generator[LocalMove]: 
      libraries = list(self.libraries)
      
      # Swap Libraries 
      for i in range(len(libraries)): 
        for j in range(i, len(libraries)): 
          if i != j:
            yield LocalMove(self.problem, libraries[i], libraries[j], swap=True)  
            yield LocalMove(self.problem, libraries[i], libraries[j], reverse=True)
              
     # Change Library 
      for i in range(self.problem.l):
        if i not in self.libraries:
          for j in self.libraries:
            if self.__can_insert(i, j):
              yield LocalMove(self.problem, i, j)
     
    def enum_local_move_lcg(self: Solution) -> Generator[LocalMove]:
      libraries =  list(self.libraries)
      swaps = 2 * (len(libraries) ** 2)
      changes = self.problem.l * len(libraries)
     
      for move in self.__non_repeating_lcg(swaps + changes):
        if move < swaps:
          i, move = libraries[move // (len(libraries) * 2)], move % (len(libraries) * 2)
          j, move = libraries[move // 2], move % 2
          if i != j: 
            if move == 0:
              yield LocalMove(self.problem, i, j, swap=True)  
            else:
              yield LocalMove(self.problem, i, j, reverse=True)
          continue
        move -= swaps
      
        i, j = move // len(self.libraries), libraries[move % len(self.libraries)] 
        if i not in self.libraries and self.__can_insert(i, j):
          yield LocalMove(self.problem, i, j) 
          
    def enum_random_local_move_wor(self: Solution) -> Generator[LocalMove]:
      for move in self.enum_local_move_lcg():
        yield move

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
      return next(self.enum_local_move_lcg()) 
      
    def step(self: Solution, move: LocalMove) -> None:   
      if move.reverse: 
        self._reverse(move.i, move.j)
      elif move.swap:
        self._swap(move.i, move.j)
      else:
        self._insert(move.i, move.j)
   
      # self.objv, self.used, self.books, self.quota = self.__heuristic_books(self.start)
      # self.objv, self.used, self.books, self.quota, self.freq = self.__heuristic_books_v2(self.start, self.freq)
      self.objv, self.used, self.books, self.quota, self.freq = self.assign_books_optimally(dict(zip(range(self.problem.b), list(map(len, self.problem.libraries)))))
      # print("AFTER", move, self.start, self.order, self.iorder, self.used)
 
    def perturb(self: Solution, kick: int) -> None:
      for _ in range(kick):
        self.step(self.random_local_move())

    def add(self: Solution, c: Component) -> None: 
      # Add Component
      score = self.__add(c)

      # Update Objective Value
      self.objv += score 
             
    def remove(self: Solution, c: Component) -> None:
      # Remove Component
      score = self.__remove(c)
      
      # Update Objective Value
      self.objv -= score         
      
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
      start, freq = self.start.copy(), copy.deepcopy(self.freq)
      
      G = self.g.copy(as_view=False)
      
      if m.swap:
        print("SWAP")
        i, j = (m.j, m.i) if self.iorder[m.i] > self.iorder[m.j] else (m.i, m.j) 
        start[i] = start[i] - self.problem.signup[i] + self.problem.signup[j]
        for x in range(self.iorder[i] + 1, self.iorder[j]):
          start[self.order[x]] = start[self.order[x - 1]] + self.problem.signup[self.order[x]]
        start[j] = start[self.order[self.iorder[j] - 1]] + self.problem.signup[i] 
        start[i], start[j] = start[j], start[i]
         
        for l in self.libraries:
          self.g[self.glibraries[l]][self.target]["capacity"] = self.problem.rate[l] * (self.problem.d - start[l])
          
        objv = -nx.cost_of_flow(self.g, nx.max_flow_min_cost(self.g, self.source, self.target))  
        
        for l in self.libraries:
          self.g[self.glibraries[l]][self.target]["capacity"] = self.problem.rate[l] * (self.problem.d - self.start[l])
        
      elif m.reverse:
        print("REV")
        i, j = (m.j, m.i) if self.iorder[m.i] > self.iorder[m.j] else (m.i, m.j) 
        start[j] = start[i] - self.problem.signup[i] + self.problem.signup[j]
        for x in range(self.iorder[j] - 1, self.iorder[i] - 1, -1):
          start[self.order[x]] = start[self.order[x + 1]] + self.problem.signup[self.order[x]]
          
        for l in self.libraries:
          self.g[self.glibraries[l]][self.target]["capacity"] = self.problem.rate[l] * (self.problem.d - start[l])
          
        objv = -nx.cost_of_flow(self.g, nx.max_flow_min_cost(self.g, self.source, self.target))  
        
        for l in self.libraries:
          self.g[self.glibraries[l]][self.target]["capacity"] = self.problem.rate[l] * (self.problem.d - self.start[l])
          
      else:
        print("INSERT")
        add, remove = m.i, m.j
        self.start[add] = self.start[remove] - self.problem.signup[remove] + self.problem.signup[add]
        s = self.start[add]
        for x in range(self.iorder[remove] + 1, len(self.order)): 
          self.start[self.order[x]] = s + self.problem.signup[self.order[x]]
          s = self.start[self.order[x]]
        self.start[remove] = self.problem.d 
        
        self.g.remove_node(self.glibraries[remove])  
        self.glibraries[add] = self.glibraries[remove] 
        self.glibraries.pop(remove)
       
        for book in self.problem.books[add]:
          if book not in self.gbooks:
            self.gbooks[book] = len(self.gbooks)
          self.g.add_edge(self.gbooks[book], self.glibraries[add], weight=-self.problem.scores[book], capacity=1)
        self.g.add_edge(self.glibraries[add], self.target, weight=0, capacity=self.problem.rate[add] * (self.problem.d - start[add]))
          
        objv = -nx.cost_of_flow(self.g, nx.max_flow_min_cost(self.g, self.source, self.target))  
        
        self.g.remove_node(self.glibraries[add])
        self.g.remove_nodes_from(list(nx.isolates(self.g)))
        
        self.glibraries[remove] = self.glibraries[add] 
        self.glibraries.pop(add)
        
        for book in self.problem.books[remove]:
          self.g.add_edge(self.gbooks[book], self.glibraries[remove], weight=-self.problem.scores[book], capacity=1)
        self.g.add_edge(self.glibraries[remove], self.target, weight=0, capacity=self.problem.rate[remove] * (self.problem.d - start[remove]))
      
      assert G.edges() == self.g.edges(), (len(self.g.edges()), len(G.edges()))
      assert G.nodes() == self.g.nodes() 
      # return self.__heuristic_books(start)[0] - self.objv 
      # return self.__heuristic_books_v2(start, freq)[0] - self.objv
      return objv - self.objv

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
        self.iorder[c.library] = len(self.order)
        self.order.append(c.library)
        
        # Update Library Start Day and Quota 
        self.day += self.problem.signup[c.library] 
        self.start[c.library] = self.day  
        self.quota[c.library] = (self.problem.d - self.day) * self.problem.rate[c.library]            
      else:
        # Update Score and Add Book 
        self.books[c.library].add(c.book)
        
        self.freq[c.book] -= 1
        
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
        self.order.remove(c.library)
        self.iorder.pop(c.library)
        
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
        
        self.freq[c.book] += 1
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
     
    def __forbidden(self: Solution, book: int, library):
      return book in self.used and library != self.used[book] 
    
    def __can_insert(self: Solution, add: int, remove: int) -> bool:
      return (self.day + self.problem.signup[add] - self.problem.signup[remove]) < self.problem.d
    
    def __heuristic_books(self: Solution, start: list[int]) -> int:
      score, used, quota = 0, dict(), [self.problem.rate[i] * (self.problem.d - start[i]) for i in range(self.problem.l)]
      books = copy.deepcopy(self.books)
      for library in self.libraries: 
        for book in self.problem.sbooks[library]: 
          if book not in used and quota[library] > 0: 
            used.setdefault(book, library)
            books[library].add(book)
            quota[library] -= 1
            score += self.problem.scores[book]
      return score, used, books, quota
   
    def __heuristic_books_v2(self: Solution, start: list[int], freq: dict[int, int]) -> None:
      
      score, used, quota = 0, dict(), [self.problem.rate[i] * (self.problem.d - start[i]) for i in range(self.problem.l)]
      books = copy.deepcopy(self.books)
      for book in sorted(set(range(self.problem.b)).symmetric_difference(used.keys()), 
                         key = lambda b: self.problem.scores[b] / freq[b] if freq[b] != 0 else 0,
                         reverse=False):
        for library in self.problem.libraries[book]:
          if book not in books[library] and quota[library] > 0:
            used.setdefault(book, library)
            books[library].add(book)
            quota[library] -= 1
            freq[book] -= 1
            score += self.problem.scores[book]
      return score, used, books, quota, freq
    
    def _insert(self: Solution, add: int, remove: int) -> None:
      # Signup Time
      self.start[add] = self.start[remove] - self.problem.signup[remove] + self.problem.signup[add]
      s = self.start[add]
      for x in range(self.iorder[remove] + 1, len(self.order)): 
        self.start[self.order[x]] = s + self.problem.signup[self.order[x]]
        s = self.start[self.order[x]]
      self.start[remove] = self.problem.d 
      
      # Order
      self.libraries.remove(remove) 
      self.libraries.add(add)
      
      self.order[self.iorder[remove]] = add
      self.iorder[add] = self.iorder[remove]
      self.iorder.pop(remove)
     
      self.g.remove_node(self.glibraries[remove])
      self.g.remove_nodes_from(list(nx.isolates(self.g)))
      
      self.glibraries[add] = self.glibraries[remove] 
      self.glibraries.pop(remove)
      
      for book in self.problem.books[add]:
        if book not in self.gbooks:
          self.gbooks[book] = len(self.gbooks)
        self.g.add_edge(self.gbooks[book], self.glibraries[add], weight=-self.problem.scores[book], capacity=1)
      self.g.add_edge(self.glibraries[add], self.target, weight=0, capacity=self.problem.rate[add] * (self.problem.d - self.start[add]))
    
      assert all(i <= self.problem.d for i in self.start), (self.start)
    
    def _swap(self: Solution, i: int, j: int) -> None:
      i, j = (j, i) if self.iorder[i] > self.iorder[j] else (i, j) 
      self.start[i] = self.start[i] - self.problem.signup[i] + self.problem.signup[j]
      for x in range(self.iorder[i] + 1, self.iorder[j]):
        self.start[self.order[x]] = self.start[self.order[x - 1]] + self.problem.signup[self.order[x]]
      self.start[j] = self.start[self.order[self.iorder[j] - 1]] + self.problem.signup[i]
      self.start[i], self.start[j] = self.start[j], self.start[i]
        
      # Order 
      self.order[self.iorder[i]], self.order[self.iorder[j]] = self.order[self.iorder[j]], self.order[self.iorder[i]]
      self.iorder[i], self.iorder[j] = self.iorder[j], self.iorder[i] 
      
      for l in self.libraries:
        self.g[self.glibraries[l]][self.target]["capacity"] = self.problem.rate[l] * (self.problem.d - self.start[l])
      
    def _reverse(self: Solution, i: int, j: int) -> None:
      i, j = (j, i) if self.iorder[i] > self.iorder[j] else (i, j) 
      
      # Signup Time and Order
      self.start[j] = self.start[i] - self.problem.signup[i] + self.problem.signup[j]
      for x in range(self.iorder[j] - 1, self.iorder[i] - 1, -1): 
        self.start[self.order[x]] = self.start[self.order[x + 1]] + self.problem.signup[self.order[x]]
        
      # Order
      ii, jj = self.iorder[i], self.iorder[j]
      for x in range((jj - ii + 1) // 2): 
        self.order[ii + x], self.order[jj - x] = self.order[jj - x], self.order[ii + x]
        self.iorder[self.order[ii + x]], self.iorder[self.order[jj - x]] = self.iorder[self.order[jj - x]], self.iorder[self.order[ii + x]]
        
      for l in self.libraries:
        self.g[self.glibraries[l]][self.target]["capacity"] = self.problem.rate[l] * (self.problem.d - self.start[l])
        
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
          
    # def _lap(self: Solution):
    #   cost, lr = [], dict()
    #   for lib in self.libraries:
    #     l = np.array([self.problem.scores[i] if i in self.problem.libraries[i] else 0 for i in range(self.problem.b)])
    #     for _ in range(min(len(self.problem.books[lib]), self.problem.rate[lib] * (self.problem.d - self.start[lib]))):
    #       cost.append(l) 
    #       lr.setdefault(len(cost) - 1, lib)
    #   t = time.perf_counter()
    #   print(len(self.libraries), self.problem.b)
    #   print(len(cost), len(cost[0]))
    #   print("Start")
    #   rows, books = optimize.linear_sum_assignment(np.array(cost))
    #   print(time.perf_counter() - t)
    #   for l, b in  zip(map(lambda x: lr[x], rows), books):
    #     self.add(Component(self.problem, b, l))
    
    def __max_flow_min_cost_graph(self: Solution, start: list[int]): 
      # Node Label Preprocessing 
      b, connections, books, libraries = 0, dict(), dict(), dict()
      for l, library in enumerate(self.order):
        libraries[library] = self.problem.b + l 
        for book in self.problem.books[library]:
          if book not in connections:
            books[book] = b
            b += 1
          connections.setdefault(book, set()).add(library)
          
      # Build Graph 
      g = nx.DiGraph()
      source = self.problem.b + self.problem.l 
      target = source + 1
      
      for book, libs in connections.items():
        g.add_edge(source, books[book], weight=0, capacity=1)
        for l in libs:
          g.add_edge(books[book], libraries[l], weight=-self.problem.scores[book], capacity=1) 
      for l in self.order:
        g.add_edge(libraries[l], target, weight=0, capacity=self.problem.rate[l] * (self.problem.d - start[l]))
      
      return g, books, libraries, source, target
    
    def assign_books_optimally(self: Solution, freq: list[int], init: bool = False): 
      if init: 
        self.g, self.gbooks, self.glibraries, self.source, self.target = self.__max_flow_min_cost_graph(self.start) 
      
      mfmc = nx.max_flow_min_cost(self.g, self.source, self.target)
      score, used, quota = 0, dict(), [self.problem.rate[i] * (self.problem.d - self.start[i]) for i in range(self.problem.l)] 
      books = [set() for _ in range(self.problem.l)]
      
      for b in self.gbooks:
        for l in self.glibraries:
          if self.gbooks[b] in mfmc and mfmc.get(self.gbooks[b]).get(self.glibraries[l]):
            if self.g not in used and quota[l] > 0: 
              used.setdefault(b, l)
              books[l].add(b)
              quota[l] -= 1
              freq[b] -= 1
              score += self.problem.scores[b]
      return score, used, books, quota, freq
      
       
    def __repr__(self: Solution) -> str:
      s = f"quota = {self.quota}\n"
      s += f"day = {self.day}\n" 
      s += f"start = {self.start}\n"
      s += f"libraries = {self.libraries}\n"
      s += f"books = {self.books}\n"
      s += f"order = {self.order}\n"
      s += f"iorder = {self.iorder}\n"
      s += f"used = {self.used}\n"
      s += f"objv = {self.objv}\n"
      s += f"freq = {self.freq}\n"
      return s
    
    def __str__(self: Solution) -> str: 
      s = f"{len(self.libraries)}\n"
      for library in self.libraries:
        s += f"{library} {len(self.books[library])}\n"
        s += f"{' '.join(str(book) for book in self.books[library])}\n"
      return s[:-1]