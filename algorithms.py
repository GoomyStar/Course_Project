'''
This file will contain implementations of each algorithm/agent type.
'''
import random

class RandomAgent:
  def __str__(self):
    return "Random Agent"
  def getMove(self, problem):
    moves = problem.getLegalMoves()
    return moves[random.randrange(len(moves))]
  

'''
Pseudo-code for Minimax:
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

Pseudo-code for Greedy Best-first Search:
procedure GBS(start, target) is:
  mark start as visited
  add start to queue
  while queue is not empty do:
    current_node ← vertex of queue with min distance to target
    remove current_node from queue
    foreach neighbor n of current_node do:
      if n not in visited then:
        if n is target:
          return n
        else:
          mark n as visited
          add n to queue
  return failure
'''
