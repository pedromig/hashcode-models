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
          
def _lap(self: Solution):
  cost, lr = [], dict()
  for lib in self.libraries:
    l = np.array([self.problem.scores[i] if i in self.problem.libraries[i] else 0 for i in range(self.problem.b)])
    for _ in range(min(len(self.problem.books[lib]), self.problem.rate[lib] * (self.problem.d - self.start[lib]))):
      cost.append(l) 
      lr.setdefault(len(cost) - 1, lib)
  t = time.perf_counter()
  print(len(self.libraries), self.problem.b)
  print(len(cost), len(cost[0]))
  print("Start")
  rows, books = optimize.linear_sum_assignment(np.array(cost))
  print(time.perf_counter() - t)
  for l, b in  zip(map(lambda x: lr[x], rows), books):
    self.add(Component(self.problem, b, l))

      # self.objv, self.used, self.books, self.quota = self.__heuristic_books(self.start)
# self.objv, self.used, self.books, self.quota, self.freq = self.__heuristic_books_v2(self.start, self.freq)
# print("AFTER", move, self.start, self.order, self.iorder, self.used)k

  
  # for book in sorted(range(self.b), key = lambda b: self.scores[b] / len(self.libraries[b]) if len(self.libraries[b]) != 0 else 0, reverse=False):
  #   for library in self.libraries[book]:
  #     if library in solution.libraries and solution.quota[library] > 0 and book not in solution.used and book not in solution.books[library]:
  #       solution.add(Component(self, book, library))
  
  # for library in solution.libraries: 
  #   for book in self.sbooks[library]: 
  #     if solution.quota[library] > 0 and book not in solution.used: 
  #       solution.add(Component(self, book, library))

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
  
  # Max Flow Special (Source and Target) Nodes
  source = self.problem.b + self.problem.l 
  target = source + 1
 
  # Library -> Books Links 
  for book, libs in connections.items():
    g.add_edge(source, books[book], weight=0, capacity=1)
    for l in libs:
      g.add_edge(books[book], libraries[l], weight=-self.problem.scores[book], capacity=1) 
  for l in self.order:
    g.add_edge(libraries[l], target, weight=0, capacity=self.problem.rate[l] * (self.problem.d - start[l]))
  
  return g, books, libraries, source, target

def assign_books_optimally(self: Solution, init: bool = False): 
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
          score += self.problem.scores[b]
          
  if init:
    self.objv, self.used, self.books, self.quota = score, used, books, quota
    
  return score, used, books, quota


