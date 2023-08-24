from __future__ import annotations

from typing import Optional, Iterable
from dataclasses import dataclass

import random
import copy
import math

import networkx as nx

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
            sbooks[lib] = tuple(sorted(self.books[lib], key = lambda b: self.scores[b], reverse=True))
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
    def __init__(self: Solution, problem: Problem, books: Optional[list[set[int]]] = None,
                 quota: Optional[list[int]] = None, used: Optional[dict[int]] = None,
                 hused: Optional[dict[int]] = None, libraries: Optional[set[int]] = None,
                 order: Optional[list[int]] = None, iorder: Optional[dict[int]] = None,
                 day: Optional[int] = None, start: Optional[list[int]] = None, 
                 objv: Optional[int] = None, complete: Optional[bool] = None) -> None:
      # Problem
      self.problem = problem

      # Solution

      ## Quota
      self.quota = quota if quota is not None else [0] * self.problem.l # Available Quota / Library
        
      ## Signup 
      self.day = day if day is not None else 0 # Last signup
      self.start = start if start is not None else [self.problem.d] * self.problem.l # Start Day / library
      
      # Max Flow Min Cost 
      self.g = nx.DiGraph()
      self.gbooks = dict(zip(range(self.problem.b), range(self.problem.b)))
      self.glibraries = dict(zip(range(self.problem.l), range(self.problem.b, self.problem.b + self.problem.l))) 
      self.source = self.problem.b + self.problem.l
      self.target = self.source + 1
      self.g.add_nodes_from((self.source, self.target))
      
      ## Libraries
      self.libraries = libraries if libraries is not None else set() # Signed Libraries
      self.order = order if order is not None else list() # Library Signup Order
      self.iorder = iorder if iorder is not None else dict() # Library Signup Order (Inverse Permutation)

      # Books 
      self.books = books if books is not None else [set() for _ in range(self.problem.l)] # Books / Library 
      self.used = used if used is not None else dict() # Used Books
      self.hused = hused if hused is not None else set() # Used Books for Heuristic
      
      # Feasibility
      self.complete = complete if complete is not None else False # Feasibility
      
      # Objective Value
      self.objv = objv if objv is not None else 0
      
    def copy(self: Solution): 
      return Solution(self.problem, copy.deepcopy(self.books), self.quota.copy(),
        copy.deepcopy(self.used), copy.deepcopy(self.hused), self.libraries.copy(), self.order.copy(),
        copy.deepcopy(self.iorder), self.day, self.start.copy(), self.objv)

    def feasible(self: Solution) -> bool:
      return self.complete

    def score(self: Solution) -> int:
      return self.objv

    def objective_value(self: Solution) -> int:
      return self.objv
    
    def enum_add_move(self: Solution) -> Generator[Component]:
      complete = True 
      for library in range(self.problem.l):
        if self.__signable(library):
          yield Component(self.problem, library)
          complete = False
      if complete:
        self.__assign_books_optimally()
          
    def enum_heuristic_add_move(self: Solution) -> Generator[Component]:      
      l = map(lambda c: (self.heuristic_value(c), c), self.enum_add_move())
      for _, c in sorted(l, key = operator.itemgetter(0), reverse=True):
        yield c
   
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
     
    def enum_random_local_move(self: Solution) -> Generator[LocalMove]:
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
      for move in self.enum_random_local_move():
        yield move
        
    def random_add_move(self: Solution) -> Component:
      libraries = list(set(self.problem.l).symmetric_difference(self.libraries))
      random.shuffle(libraries)
      for library in libraries:
        if self.__signable(library):
          return Component(self.problem, library)
 
    def random_remove_move(self: Solution) -> Component:
      libraries = list(self.libraries)
      random.shuffle(libraries)
      for library in libraries:
        return Component(self.problem, library)
        
    def random_local_move(self: Solution) -> LocalMove:
      return next(self.enum_random_local_move()) 
      
    def step(self: Solution, move: LocalMove) -> None:   
      if move.reverse: 
        print("REVERSE")
        self._reverse(move.i, move.j)
      elif move.swap:
        print("SWAP")
        self._swap(move.i, move.j)
      else:
        print("INSERT")
        self._insert(move.i, move.j)
      self.__assign_books_optimally()
 
    def perturb(self: Solution, kick: int) -> None:
      for _ in range(kick):
        self.step(self.random_local_move())

    def add(self: Solution, c: Component) -> None: 
      # Add Library
      self.libraries.add(c.library)
      self.iorder[c.library] = len(self.order)
      self.order.append(c.library)
      
      # Update Library Start Day and Quota 
      self.day += self.problem.signup[c.library]
      self.start[c.library] = self.day  
      self.quota[c.library] = (self.problem.d - self.day) * self.problem.rate[c.library]
    
      # Update used books in heuristic 
      books = [b for b in self.problem.sbooks[c.library] if b not in self.hused][:self.quota[c.library]] 
      for b in books:  
        self.hused.add(b)
    
      # Update Max Flow Min Cost Graph 
      for b in self.problem.books[c.library]:
        if self.gbooks[b] not in self.g.nodes():
          self.g.add_edge(self.source, self.gbooks[b], weight=0, capacity=1) 
        self.g.add_edge(self.gbooks[b], self.glibraries[c.library], weight=-self.problem.scores[b], capacity=1)
      self.g.add_edge(self.glibraries[c.library], self.target, weight=0, capacity=self.quota[c.library])
      
    def remove(self: Solution, c: Component) -> None:
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
       
    def heuristic_value(self: Solution, c: Component):
      return sum([self.problem.scores[b] for b in self.problem.sbooks[c.library] if b not in self.hused][:(self.problem.d - self.day - self.problem.signup[c.library]) * self.problem.rate[c.library]]) / self.problem.signup[c.library]
     
    def objective_increment_local(self: Solution, m: LocalMove) -> int:
      start = self.start.copy()
      if m.swap:
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
        add, remove = m.i, m.j
        start[add] = start[remove] - self.problem.signup[remove] + self.problem.signup[add]
        s = start[add]
        for x in range(self.iorder[remove] + 1, len(self.order)): 
          start[self.order[x]] = s + self.problem.signup[self.order[x]]
          s = start[self.order[x]]
        start[remove] = self.problem.d 
        
        self.libraries.add(add)
        self.libraries.remove(remove)
         
        self.g.remove_node(self.glibraries[remove])
        self.g.remove_nodes_from(list(nx.isolates(self.g)))
         
        for b in self.problem.books[add]:
          if self.gbooks[b] not in self.g.nodes():
            self.g.add_edge(self.source, self.gbooks[b], weight=0, capacity=1) 
          self.g.add_edge(self.gbooks[b], self.glibraries[add], weight=-self.problem.scores[b], capacity=1)
        self.g.add_edge(self.glibraries[add], self.target, weight=0, capacity=self.problem.rate[add] * (self.problem.d - start[add]))
          
        for l in self.libraries:
          self.g[self.glibraries[l]][self.target]["capacity"] = self.problem.rate[l] * (self.problem.d - start[l])
         
        objv = -nx.cost_of_flow(self.g, nx.max_flow_min_cost(self.g, self.source, self.target))  
        
        self.libraries.add(remove)
        self.libraries.remove(add)
        
        self.g.remove_node(self.glibraries[add])
        self.g.remove_nodes_from(list(nx.isolates(self.g)))
        
        for b in self.problem.books[remove]: 
          if self.gbooks[b] not in self.g.nodes():
            self.g.add_edge(self.source, self.gbooks[b], weight=0, capacity=1) 
          self.g.add_edge(self.gbooks[b], self.glibraries[remove], weight=-self.problem.scores[b], capacity=1)
        self.g.add_edge(self.glibraries[remove], self.target, weight=0, capacity=self.problem.rate[remove] * (self.problem.d - self.start[remove]))  
        
        for l in self.libraries:
          self.g[self.glibraries[l]][self.target]["capacity"] = self.problem.rate[l] * (self.problem.d - self.start[l])
        
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
                 
    def __signed(self: Solution, library: int) -> bool:
      return library in self.libraries

    def __signable(self: Solution, library: int) -> bool:
      return not self.__signed(library) and (self.day + self.problem.signup[library]) < self.problem.d
     
    def __can_insert(self: Solution, add: int, remove: int) -> bool:
      return (self.day + self.problem.signup[add] - self.problem.signup[remove]) < self.problem.d
     
    def _swap(self: Solution, i: int, j: int) -> None:
      i, j = (j, i) if self.iorder[i] > self.iorder[j] else (i, j)
      
      self.start[i] = self.start[i] - self.problem.signup[i] + self.problem.signup[j]
      for x in range(self.iorder[i] + 1, self.iorder[j]):
        self.start[self.order[x]] = self.start[self.order[x - 1]] + self.problem.signup[self.order[x]]
      self.start[j] = self.start[self.order[self.iorder[j] - 1]] + self.problem.signup[i]
      self.start[i], self.start[j] = self.start[j], self.start[i]
        
      self.order[self.iorder[i]], self.order[self.iorder[j]] = self.order[self.iorder[j]], self.order[self.iorder[i]]
      self.iorder[i], self.iorder[j] = self.iorder[j], self.iorder[i] 
      
      for l in self.libraries:
        self.g[self.glibraries[l]][self.target]["capacity"] = self.problem.rate[l] * (self.problem.d - self.start[l])
        
    def _reverse(self: Solution, i: int, j: int) -> None:
      i, j = (j, i) if self.iorder[i] > self.iorder[j] else (i, j) 
      
      self.start[j] = self.start[i] - self.problem.signup[i] + self.problem.signup[j]
      for x in range(self.iorder[j] - 1, self.iorder[i] - 1, -1): 
        self.start[self.order[x]] = self.start[self.order[x + 1]] + self.problem.signup[self.order[x]]
        
      ii, jj = self.iorder[i], self.iorder[j]
      for x in range((jj - ii + 1) // 2): 
        self.order[ii + x], self.order[jj - x] = self.order[jj - x], self.order[ii + x]
        self.iorder[self.order[ii + x]], self.iorder[self.order[jj - x]] = self.iorder[self.order[jj - x]], self.iorder[self.order[ii + x]]
        
      for l in self.libraries:
        self.g[self.glibraries[l]][self.target]["capacity"] = self.problem.rate[l] * (self.problem.d - self.start[l])
        
    def _insert(self: Solution, add: int, remove: int) -> None:
      self.start[add] = self.start[remove] - self.problem.signup[remove] + self.problem.signup[add]
      s = self.start[add]
      for x in range(self.iorder[remove] + 1, len(self.order)): 
        self.start[self.order[x]] = s + self.problem.signup[self.order[x]]
        s = self.start[self.order[x]]  
      self.day = self.start[self.order[-1]] 
      self.start[remove] = self.problem.d 
      
      self.libraries.add(add)
      self.libraries.remove(remove) 
      
      self.order[self.iorder[remove]] = add
      self.iorder[add] = self.iorder[remove]
      self.iorder.pop(remove)
    
      self.g.remove_node(self.glibraries[remove])
      self.g.remove_nodes_from(list(nx.isolates(self.g)))
      
      for b in self.problem.books[add]: 
        if self.gbooks[b] not in self.g.nodes():
          self.g.add_edge(self.source, self.gbooks[b], weight=0, capacity=1) 
        self.g.add_edge(self.gbooks[b], self.glibraries[add], weight=-self.problem.scores[b], capacity=1)
      self.g.add_edge(self.glibraries[add], self.target, weight=0, capacity=self.problem.rate[add] * (self.problem.d - self.start[add]))
      
      for l in self.libraries:
        self.g[self.glibraries[l]][self.target]["capacity"] = self.problem.rate[l] * (self.problem.d - self.start[l])
        
      # FIXME: Add remaining libraries? How
        
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
          
    def __assign_books_optimally(self: Solution) -> None:
      mfmc = nx.max_flow_min_cost(self.g, self.source, self.target)
      
      score, used, quota = 0, dict(), [self.problem.rate[i] * (self.problem.d - self.start[i]) for i in range(self.problem.l)] 
      books = [set() for _ in range(self.problem.l)]
      
      for l in self.libraries:
        for b in self.problem.books[l]:
          if mfmc.get(self.gbooks[b]).get(self.glibraries[l]):
            used.setdefault(b, l)
            books[l].add(b)
            quota[l] -= 1
            score += self.problem.scores[b]
            
      self.complete = True
      self.objv, self.used, self.books, self.quota = score, used, books, quota
      
    def __eval(self: Solution):
      return - nx.cost_of_flow(self.g, nx.max_flow_min_cost(self.g, self.source, self.target))
          
    def __surrogate(self: Solution, start: list[int]) -> int:
      score, used, quota = 0, dict(), [self.problem.rate[i] * (self.problem.d - start[i]) for i in range(self.problem.l)]
      for library in self.order:
        for book in self.problem.sbooks[library]: 
          if book not in used and quota[library] > 0: 
            used.setdefault(book, library)
            quota[library] -= 1
            score += self.problem.scores[book]
      return score
   
    def __surrogate_v2(self: Solution, start: list[int]) -> None:
      score, used, quota = 0, dict(), [self.problem.rate[i] * (self.problem.d - start[i]) for i in range(self.problem.l)]
      freq = dict(map(lambda b: (b, len(self.problem.libraries[b])), range(self.problem.b)))
      
      for book in sorted(set(range(self.problem.b)).symmetric_difference(used.keys()), 
                        key = lambda b: self.problem.scores[b] / freq[b] if freq[b] != 0 else 0,
                        reverse=False):
        for library in self.problem.libraries[book]:
          if quota[library] > 0:
            used.setdefault(book, library)
            quota[library] -= 1
            freq[book] -= 1
            score += self.problem.scores[book]
      return score
   
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
      return s
    
    def __str__(self: Solution) -> str: 
      s = f"{len(self.libraries)}\n"
      for library in self.libraries:
        s += f"{library} {len(self.books[library])}\n"
        s += f"{' '.join(str(book) for book in self.books[library])}\n"
      return s[:-1]

class Component:
  def __init__(self: Component, problem: Problem, library: int) -> None:
    self.problem = problem
    self.library = library 
    
  def __lt__(self: Component, other: Component) -> bool:
      return self.library < other.library

  def __eq__(self: Component, other: Component) -> bool:
      return self.library == other.library
     
  def __repr__(self: Component):
    return self.__str__()
    
  def __str__(self: Component):
    return f"Component({self.library})"

class LocalMove: 
  def __init__(self: LocalMove, problem: Problem, 
               i: tuple[int] | None = None, 
               j: tuple[int] | None = None,
               reverse: bool | None = None,
               swap: bool | None = None) -> None:
    self.problem = problem
    self.i, self.j = i, j
    self.reverse = reverse 
    self.swap = swap

  def __repr__(self: LocalMove) -> str:
    return self.__str__() 
    
  def __str__(self: LocalMove) -> str:
    return f"({self.i}, {self.j}, {self.swap}, {self.reverse})"