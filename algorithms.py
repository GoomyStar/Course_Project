'''
This file will contain implementations of each algorithm/agent.


Pseudocode for Minimax:
function minimax(node, depth, maximizingPlayer) is
    if depth = 0 or node is a terminal node then
        return the heuristic value of node
    if maximizingPlayer then
        value := −∞
        for each child of node do
            value := max(value, minimax(child, depth − 1, FALSE))
        return value
    else (* minimizing player *)
        value := +∞
        for each child of node do
            value := min(value, minimax(child, depth − 1, TRUE))
        return value


Pseudo-code for Alpha-beta pruning:
function alphabeta(node, depth, α, β, maximizingPlayer) is
    if depth == 0 or node is terminal then
        return the heuristic value of node
    if maximizingPlayer then
        value := −∞
        for each child of node do
            value := max(value, alphabeta(child, depth − 1, α, β, FALSE))
            if value ≥ β then
                break (* β cutoff *)
            α := max(α, value)
        return value
    else
        value := +∞
        for each child of node do
            value := min(value, alphabeta(child, depth − 1, α, β, TRUE))
            if value ≤ α then
                break (* α cutoff *)
            β := min(β, value)
        return value

Pseudocode for Greedy Best-First Search:
procedure GBS(start, target) is:
  mark start as visited
  add start to queue
  while queue is not empty do:
    current_node ← vertex of queue with min distance to target
    remove current_node from queue
    for each neighbor n of current_node do:
      if n not in visited then:
        if n is target:
          return n
        else:
          mark n as visited
          add n to queue
  return failure

How the pseudo-code maps onto the classes below
  minimax    -> MinimaxAgent._minimax      (alphaBeta=False)
  alphabeta  -> MinimaxAgent._alphabeta    (alphaBeta=True)
  GBS        -> GreedyBestFirstAgent.chooseMove: "distance to target" is the heuristic
                (the queue is a heap ordered by it), a neighbor is a successor state,
                and the target is a terminal state the root player has won.
                If the node budget runs out first, it plays the best-looking root move
                instead of returning "failure".

Agents:
------
RandomAgent             (Given)
MinimaxAgent            depth-limited Minimax (Lecture VIII / Minimax Alpha-beta pruning),
GreedyBestFirstAgent    Greedy Best-First Search (Lecture VII / Wikipedia "Best-first search")
ApproximateQAgent       (Work in Progress)
'''

import heapq
import math
import random
import time

WIN_SCORE = 1e6        # terminal values dwarf any heuristic value
EPS = 1e-9


class RandomAgent:
  ''' Given class, but I modified it with the game's movement heuristics. '''
  def __init__(self):
    self.moves = 0
    self.totalTime = 0.0
  def __str__(self):
    return "Random Agent"
  def getMove(self, problem):
    t = time.perf_counter()
    moves = problem.getLegalMoves()
    move = moves[random.randrange(len(moves))]
    self.moves += 1
    self.totalTime += time.perf_counter() - t
    return move


class TimeAgent:
  '''Records decision time and nodes per move, so comparisons can report how long each
  approach takes to decide (this does not change the outcome of a game).'''
  def __init__(self):
    self.moves = 0
    self.totalTime = 0.0
    self.totalNodes = 0
    self.lastTime = 0.0
    self.nodes = 0
  def getMove(self, problem):
    self.nodes = 0
    t = time.perf_counter()
    move = self.chooseMove(problem)
    self.lastTime = time.perf_counter() - t
    self.moves += 1
    self.totalTime += self.lastTime
    self.totalNodes += self.nodes
    return move
  def avgMoveTime(self):
    if self.moves == 0:
      return 0.0
    return self.totalTime / self.moves
  def resetStats(self):
    self.moves = 0
    self.totalTime = 0.0
    self.totalNodes = 0


class HeuristicEvaluator:
  '''evaluate(problem, state, player) = sum_k weight_k * feature_k(state, player).
  Default weights are the problem's hand-picked HEURISTIC_WEIGHTS for the feature set.'''
  def __init__(self, featureSet='shared', weights=None):
    self.featureSet = featureSet
    self.weights = weights
  def evaluate(self, problem, state, player):
    if self.weights is not None:
      w = self.weights
    else:
      w = problem.weightsFor(self.featureSet)
    f = problem.getFeatures(state, player, self.featureSet)
    total = 0
    for k in f:
      total = total + w.get(k, 0.0) * f[k]
    return total


class MinimaxAgent(TimeAgent):
  '''Minimax with Alpha-Beta pruning:

      value(state, depth):
        terminal         -> utility * (WIN_SCORE + depth)   (prefer faster wins)
        depth == 0       -> heuristic evaluation of the state for the root player
        player to move is the root player (MAX) -> max over children
        otherwise (MIN)                         -> min over children

  Who is MAX/MIN is decided by problem.getCurrentPlayer(state), not by ply parity, so
  Mancala's extra turns are handled correctly.

  Alpha-Beta pruning: returns exactly the same value as plain Minimax but skips
  branches that cannot change the result: a MAX node cuts when value >= beta, a MIN node
  cuts when value <= alpha.

  ordering (teacher's move-ordering modes; only used with Alpha-Beta):
      'heuristic'  best child first for the side to move (most cutoffs)
      'off'        the problem's natural move order
      'reversed'   worst child first (fewest cutoffs)
  randomTies: choose randomly among equally good root moves, so games are not identical.'''
  def __init__(self, depth=3, alphaBeta=True, ordering='heuristic', featureSet='shared',
               evaluator=None, randomTies=True, seed=None):
    super().__init__()
    self.depth = depth
    self.alphaBeta = alphaBeta
    self.ordering = ordering
    if evaluator is None:
      self.evaluator = HeuristicEvaluator(featureSet)
    else:
      self.evaluator = evaluator
    self.randomTies = randomTies
    self.rng = random.Random(seed)
  def __str__(self):
    name = "Minimax"
    if self.alphaBeta:
      name = name + "+AlphaBeta"
    return name + "(depth=" + str(self.depth) + ")"

  def _children(self, problem, state, me, maximizing):
    kids = []
    for m in problem.getLegalMoves(state):
      kids.append((m, problem.getSuccessor(m, state)))
    if self.alphaBeta and self.ordering != 'off' and len(kids) > 1:
      # score every child, then sort them with a simple insertion sort
      # (children with equal scores stay in their original order)
      scores = []
      for kid in kids:
        scores.append(self.evaluator.evaluate(problem, kid[1], me))
      highFirst = (maximizing == (self.ordering == 'heuristic'))
      sortedKids = []
      sortedScores = []
      for i in range(len(kids)):
        position = len(sortedKids)
        while position > 0:
          if highFirst and sortedScores[position - 1] < scores[i]:
            position = position - 1
          elif (not highFirst) and sortedScores[position - 1] > scores[i]:
            position = position - 1
          else:
            break
        sortedKids.insert(position, kids[i])
        sortedScores.insert(position, scores[i])
      kids = sortedKids
    return kids
  def _leaf(self, problem, state, depth, me):
    self.nodes += 1
    if problem.isTerminal(state):
      return problem.utility(me, state) * (WIN_SCORE + depth)
    return self.evaluator.evaluate(problem, state, me)

  def _minimax(self, problem, state, depth, me):
    if depth <= 0 or problem.isTerminal(state):
      return self._leaf(problem, state, depth, me)
    self.nodes += 1
    maximizing = problem.getCurrentPlayer(state) == me
    vals = []
    for kid in self._children(problem, state, me, maximizing):
      s = kid[1]
      vals.append(self._minimax(problem, s, depth - 1, me))
    if maximizing:
      return max(vals)
    return min(vals)

  def _alphabeta(self, problem, state, depth, alpha, beta, me):
    if depth <= 0 or problem.isTerminal(state):
      return self._leaf(problem, state, depth, me)
    self.nodes += 1
    if problem.getCurrentPlayer(state) == me:                    # MAX
      value = -math.inf
      for kid in self._children(problem, state, me, True):
        child = kid[1]
        value = max(value, self._alphabeta(problem, child, depth - 1, alpha, beta, me))
        alpha = max(alpha, value)
        if value >= beta:
          break                                                  # beta cutoff
      return value
    value = math.inf                                             # MIN
    for kid in self._children(problem, state, me, False):
      child = kid[1]
      value = min(value, self._alphabeta(problem, child, depth - 1, alpha, beta, me))
      beta = min(beta, value)
      if value <= alpha:
        break                                                    # alpha cutoff
    return value

  def chooseMove(self, problem):
    state = problem.state
    me = problem.getCurrentPlayer(state)
    best = -math.inf
    bestMoves = []
    alpha = -math.inf
    for kid in self._children(problem, state, me, True):
      move = kid[0]
      child = kid[1]
      if self.alphaBeta:
        v = self._alphabeta(problem, child, self.depth - 1, alpha, math.inf, me)
      else:
        v = self._minimax(problem, child, self.depth - 1, me)
      if v > best + EPS:
        best = v
        bestMoves = [move]
      elif abs(v - best) <= EPS:
        bestMoves.append(move)
      # keep the window just below best so moves tied with it are still searched exactly
      if self.randomTies:
        alpha = max(alpha, best - EPS)
      else:
        alpha = max(alpha, best)
    self.lastValue = best
    if self.randomTies:
      return self.rng.choice(bestMoves)
    return bestMoves[0]

class GreedyBestFirstAgent(TimeAgent):
  '''Greedy Best-First Search: like uniform-cost search, but the priority queue is ordered
  by the heuristic ALONE (path cost is ignored), so it is complete on finite graphs but not
  optimal.

    goal    : a terminal state won by the player to move at the root
    h(node) : -evaluate(node, mover), mover = the player who just moved into `node`
              (lower = more promising for that mover, so each side greedily expands
              the lines that look best for itself)
    result  : the root move leading to the first goal popped; if the node budget runs out
              first, the root move whose child has the best heuristic value.

  evaluator: anything with evaluate(problem, state, player). Pass a trained
  ApproximateQAgent here to MIX greedy search with the learned Q-function
  (heuristic = learned weights instead of hand-picked ones).'''
  def __init__(self, nodeBudget=300, featureSet='shared', evaluator=None):
    super().__init__()
    self.nodeBudget = nodeBudget
    if evaluator is None:
      self.evaluator = HeuristicEvaluator(featureSet)
    else:
      self.evaluator = evaluator
  def __str__(self):
    return "GreedyBestFirst(budget=" + str(self.nodeBudget) + ")"

  def chooseMove(self, problem):
    state = problem.state
    me = problem.getCurrentPlayer(state)
    ev = self.evaluator.evaluate
    tie = 0
    frontier = []
    visited = set()
    visited.add(problem.stateKey(state))
    fallback = None
    fallbackVal = -math.inf

    for move in problem.getLegalMoves(state):
      child = problem.getSuccessor(move, state)
      if problem.isTerminal(child) and problem.getWinner(child) == me:
        return move                               # immediate win
      val = ev(problem, child, me)
      if val > fallbackVal:
        fallback = move
        fallbackVal = val
      key = problem.stateKey(child)
      if key not in visited:
        visited.add(key)
        mover = problem.getCurrentPlayer(state)
        heapq.heappush(frontier, (-ev(problem, child, mover), tie, child, move))
        tie = tie + 1

    while len(frontier) > 0 and self.nodes < self.nodeBudget:
      item = heapq.heappop(frontier)
      node = item[2]
      rootMove = item[3]
      self.nodes += 1
      if problem.isTerminal(node):
        if problem.getWinner(node) == me:
          return rootMove                         # goal reached
        continue
      mover = problem.getCurrentPlayer(node)
      for move in problem.getLegalMoves(node):
        child = problem.getSuccessor(move, node)
        key = problem.stateKey(child)
        if key in visited:
          continue
        visited.add(key)
        heapq.heappush(frontier, (-ev(problem, child, mover), tie, child, rootMove))
        tie = tie + 1
    return fallback


'''class AQL(TimeAgent):'''
