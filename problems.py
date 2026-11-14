'''
This file will contain your the code for your problems. 
'''

import numpy as np
import cv2

class Game:
    def __init__(self, problem, pZero, pOne,verbose=True):
      self.problem = problem      
      self.players = [pZero,pOne]   
      self.verbose = verbose   
      if self.verbose:
         self.problem.showState()               
    def playGame(self):
      pCur=0
      while self.problem.isTerminal()==False:           
         move = self.players[pCur].getMove(self.problem)           
         self.problem.doMove(move)
         if self.verbose:
            self.problem.showState()
            print(f"Move {self.problem.ticks}:")
            print(self.problem.state)
         pCur = np.abs(pCur-1)
        
      wIndex = self.problem.getWinner()
      winner = self.players[wIndex]
      if wIndex == -1:
         winner = "DRAW"
      #display final state
      print(f"The Winner is {winner} ({wIndex})!")
      if self.verbose:
         self.problem.showState(4000)
      return wIndex
 
 
class Problem:
   '''Shared helpers for every problem (see the interface in the module docstring).'''
   HEURISTIC_WEIGHTS = {}   # featureSet name -> {feature: hand-picked weight}
 
   def getCurrentPlayer(self, state=None):
      raise NotImplementedError
   def getFeatures(self, state, player, featureSet='shared'):
      raise NotImplementedError
   def weightsFor(self, featureSet='shared'):
      return self.HEURISTIC_WEIGHTS[featureSet]
   def utility(self, player, state=None):
      # +1 if player won, -1 if they lost, 0 for a draw (used by Minimax at terminal states)
      w = self.getWinner(state)
      if w == -1:
         return 0
      if w == player:
         return 1
      return -1
   def stateKey(self, state=None):
      if state is None:
         state = self.state
      # hashable key for a board, so search can remember visited states
      return state.tobytes()
   def doMove(self, move):
      self.state = self.getSuccessor(move, self.state)
      self.ticks += 1
   def showState(self, ms=1000, state=None):
      print(self.render(state))
   def render(self, state=None):
      if state is None:
         state = self.state
      return str(state)
 
 
class TicTacToe:
   def __init__(self):
      self.state = np.zeros((3,3))      
      self.ticks=-1      
   def getLegalMoves(self, state=None):
      if state is None:
         state = self.state
      moves = []
      mark = 1
      if np.sum(np.abs(state))%2!=0:
         mark = -1       
      for i in range(state.shape[0]):
         for j in range(state.shape[1]):
            if state[i,j]==0:
               moves.append((mark,(i,j)))
      return moves             
   def getSuccessor(self, move, state):
      mark = move[0]
      loc = move[1]
      state[loc] = mark
      return state
   def doMove(self,move):
      self.state = self.getSuccessor(move,self.state)
      self.ticks+=1
   def isTerminal(self, state=None):
      if state is None:
         state = self.state         
      terminal = False
      val = self.evalTerminal(state)      
      if val ==0 and np.sum(np.abs(state))==9:
         terminal = True
      if val!=0:
         terminal = True
      return terminal
   def evalTerminal(self, state=None):
      if state is None:
         state = self.state
      val = 0      
      
      pZeroWins = [np.max(np.sum(state,0))==3,np.max(np.sum(state,1))==3,np.trace(state)==3,np.trace(state[:,::-1])==3]
      pOneWins = [np.min(np.sum(state,0))==-3,np.min(np.sum(state,1))==-3,np.trace(state)==-3,np.trace(state[:,::-1])==-3]

      if np.any(pZeroWins):
         val = 1
      if np.any(pOneWins):
         val = -1
      return val
   def getWinner(self, state=None):
      if state is None:
         state = self.state
      val = self.evalTerminal(state)
      if val==1:
         return 0 
      elif val==-1:
         return 1
      else:
         return -1
   def showState(self, ms = 1000,state=None):
      if state is None:
         state = self.state  

      screen = np.zeros((150,150)).astype(np.uint8)         
      screen = cv2.line(screen,(0,49),(149,49),255) 
      screen = cv2.line(screen,(0,99),(149,99),255)
      screen = cv2.line(screen,(49,0),(49,149),255) 
      screen = cv2.line(screen,(99,0),(99,149),255)

      for i in range(state.shape[0]):
         for j in range(state.shape[1]):
            if state[i,j]==1:
               screen = cv2.line(screen,(5 +j*50,5+i*50),(44+j*50,44+i*50),255,2,cv2.LINE_AA) 
               screen = cv2.line(screen,(44+j*50,5+i*50),(5 +j*50,44+i*50),255,2,cv2.LINE_AA) 
            if state[i,j]==-1:
               screen = cv2.circle(screen, (25 +j*50,25+i*50), 20, 255, 2,cv2.LINE_AA) 
               
      cv2.imshow('TicTacToe',screen)
      cv2.waitKey(ms)
 
 
class ConnectFour(Problem):
   '''6 rows x 7 columns. state[r, c]: 0 empty, 1 = player 0, -1 = player 1; row 0 is the
   top. A move is (mark, column): the disc drops to the lowest empty row of that column.'''
   ROWS = 6
   COLS = 7
   HEURISTIC_WEIGHTS = {
      'shared': {'three': 5.0, 'two': 2.0, 'center': 3.0},
      'custom': {'three': 5.0, 'two': 2.0, 'center': 3.0, 'one': 0.5, 'inner': 1.0},
   }
 
   def __init__(self):
      # empty board
      self.state = np.zeros((self.ROWS, self.COLS))
      self.ticks = -1
      # every horizontal / vertical / diagonal window of four cells
      self.WINDOWS = []
      # directions: right, down, down-right, down-left
      directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
      for r in range(6):
         for c in range(7):
            # start from each cell and try every direction
            for direction in directions:
               dr = direction[0]
               dc = direction[1]
               endRow = r + 3 * dr
               endCol = c + 3 * dc
               # keep the window only if its last cell is still on the board
               if endRow >= 0 and endRow < 6 and endCol >= 0 and endCol < 7:
                  window = []
                  for k in range(4):
                     window.append((r + k * dr, c + k * dc))
                  self.WINDOWS.append(window)
   def getCurrentPlayer(self, state=None):
      if state is None:
         state = self.state
      # even number of discs -> player 0's turn, odd -> player 1's turn
      return int(np.sum(np.abs(state)) % 2)
   def getLegalMoves(self, state=None):
      if state is None:
         state = self.state
      # `mark` = my disc value, `-mark` = opponent's
      mark = 1
      if np.sum(np.abs(state)) % 2 != 0:
         mark = -1
      # a column is playable while its top cell is empty
      moves = []
      for c in range(self.COLS):
         if state[0, c] == 0:
            moves.append((mark, c))
      return moves
   def getSuccessor(self, move, state):
      mark = move[0]
      col = move[1]
      # copy first so the original board is never changed (search needs this)
      state = state.copy()
      row = self.ROWS - 1                       # lowest empty cell
      while state[row, col] != 0:
         row = row - 1
      state[row, col] = mark
      return state
   def _windowValues(self, state):
      # for every 4-cell window, read the values on the board
      board = state.tolist()
      values = []
      for window in self.WINDOWS:
         oneWindow = []
         for cell in window:
            oneWindow.append(board[cell[0]][cell[1]])
         values.append(oneWindow)
      return values          # (69, 4)
   def evalTerminal(self, state=None):
      if state is None:
         state = self.state
      # four of the same mark in a window = win (sum is +4 or -4)
      for window in self._windowValues(state):
         total = 0
         for value in window:
            total = total + value
         if total == 4:
            return 1
         if total == -4:
            return -1
      return 0
   def isTerminal(self, state=None):
      if state is None:
         state = self.state
      if self.evalTerminal(state) != 0:
         return True
      # not over if any column still has room; a full board with no winner is a draw
      for c in range(self.COLS):
         if state[0, c] == 0:
            return False
      return True
   def getWinner(self, state=None):
      val = self.evalTerminal(state)
      if val == 1:
         return 0
      if val == -1:
         return 1
      return -1
   def getFeatures(self, state, player, featureSet='shared'):
      mark = 1
      if player != 0:
         mark = -1
      # count windows holding 3, 2 or 1 of my discs (and none of theirs), and the same for the opponent
      w = self._windowValues(state)
      myThree = 0
      theirThree = 0
      myTwo = 0
      theirTwo = 0
      myOne = 0
      theirOne = 0
      for window in w:
         mine = 0
         theirs = 0
         for value in window:
            if value == mark:
               mine = mine + 1
            if value == -mark:
               theirs = theirs + 1
         if mine == 3 and theirs == 0:
            myThree = myThree + 1
         if theirs == 3 and mine == 0:
            theirThree = theirThree + 1
         if mine == 2 and theirs == 0:
            myTwo = myTwo + 1
         if theirs == 2 and mine == 0:
            theirTwo = theirTwo + 1
         if mine == 1 and theirs == 0:
            myOne = myOne + 1
         if theirs == 1 and mine == 0:
            theirOne = theirOne + 1
      # centre column (col 3) is the most valuable: my discs add, theirs subtract
      centerTotal = 0
      for r in range(self.ROWS):
         centerTotal = centerTotal + state[r, 3] * mark
      # scale each feature to roughly [-1, 1] so the weights are comparable
      f = {
         'three': float(myThree - theirThree) / 5.0,
         'two':   float(myTwo - theirTwo) / 10.0,
         'center': float(centerTotal) / 6.0,
      }
      # 'custom' set adds: windows with one disc, and discs in the middle columns 2-4
      if featureSet == 'custom':
         f['one'] = float(myOne - theirOne) / 20.0
         innerTotal = 0
         for r in range(self.ROWS):
            for c in range(2, 5):
               innerTotal = innerTotal + state[r, c] * mark
         f['inner'] = float(innerTotal) / 18.0
      return f
   def render(self, state=None):
      if state is None:
         state = self.state
      # text picture of the board: X = player 0, O = player 1
      sym = {0: '.', 1: 'X', -1: 'O'}
      text = ""
      for row in state:
         line = ""
         for v in row:
            if line != "":
               line = line + " "
            line = line + sym[int(v)]
         text = text + line + "\n"
      return text + "0 1 2 3 4 5 6"
 
 
class Mancala(Problem):
    '''Mancala. state is a length-15 array:
         [0..5]   player 0's pits (left to right)      [6]  player 0's store
         [7..12]  player 1's pits                       [13] player 1's store
         [14]     player to move (0 or 1)
    A move is a pit number 0..5 counted from the players's left. Seeds are placed
    counter-clockwise, skipping the opponent's store.
      * Last seed in your own store -> you move again (turns do NOT always alternate).
      * Last seed in an empty pit on your side, opposite pit non-empty -> capture both.
      * When either side has no seeds left, the other side's seeds go to its owner's
        store and the player with the larger store wins (equal stores -> draw).'''
    PITS = 6
    SEEDS = 4
    HEURISTIC_WEIGHTS = {
       'shared': {},
       'custom': {},
    }

   def __init__(self):
      # TODO: self.state = length-15 array: SEEDS in every pit, 0 in both stores,
      #       player 0 to move.  self.ticks = -1 like ConnectFour.

      pass
   def getCurrentPlayer(self, state=None):
      # TODO: read state[14] and return it as an int

      pass
   def getLegalMoves(self, state=None):
      # TODO: pit numbers 0..5 that are non-empty for the player to move

      pass
   def getSuccessor(self, move, state):
      # TODO: copy first (search relies on this), then:
      #   1. pick up the seeds, sow them one by one, skipping the opponent's store
      #   2. note where the last seed landed
      #   3. capture rule (was the pit empty before the seed landed?)
      #   4. end-of-game sweep
      #   5. set state[14] for the next turn (extra turn or flip)
      pass
   def isTerminal(self, state=None):
      pass
   def getWinner(self, state=None):
      # 0 or 1 by larger store, -1 for a draw (and -1 if the game isn't over yet)
      pass
   def getFeatures(self, state, player, featureSet='shared'):
      # TODO: return a dict with each value scaled to roughly [-1, 1]
      pass
   def render(self, state=None):
      # TODO: a text picture of the board
      pass
   # LAST: fill HEURISTIC_WEIGHTS
 
 
class Wallz(Problem):
    '''Wallz = Quoridor. Race your pawn to the opposite side of the board. On your turn
    either move your pawn one step (with Quoridor's jump rules) or place a wall that
    lengthens the opponent's path. A wall may never completely cut a pawn off from its goal.
 
    Example:
    Player 0 starts at the bottom middle and must reach row 0. Player 1 starts at the top
    middle and must reach row size-1. Defaults are the standard 9x9 board with 10 walls
    each (up to 2*8*8 = 128 wall placements plus pawn moves per turn), but size,
    walls and maxPlies are configurable so experiments can run on smaller boards.
    If maxPlies moves are made without a winner the game is a draw.'''
    
    HEURISTIC_WEIGHTS = {
        'shared': {},
        'custom': {},
    }
      def __init__(self, size=9, walls=10, maxPlies=200):
         # TODO: save settings, build state (pawns at bottom/top middle, 10 walls each,
         #       player 0 to move), self.ticks = -1
      def getCurrentPlayer(self, state=None):
         # TODO: default state, return player slot as int
      def _isBlocked(self, state, r1, c1, r2, c2):
         # TODO: True if a wall sits between two neighbouring squares
      def _pawnMoves(self, state, player):
         # TODO: up/down/left/right minus blocked (jumps later)
      def _shortestPath(self, state, player):
         # TODO: BFS to goal row; steps or None.  CHECK: empty 9x9 -> 8
      def _wallMoves(self, state, player):
         # TODO: no overlap/crossing; BOTH players must still have a path.  CHECK: opening -> 128
      def getLegalMoves(self, state=None):
         # TODO: pawn moves + wall moves.  CHECK: opening -> 131
      def getSuccessor(self, move, state):
         # TODO: copy; apply pawn move or wall (-1 wall left); plies+1; flip player
      def isTerminal(self, state=None):
         # TODO: a pawn on its goal row, or plies >= maxPlies
      def getWinner(self, state=None):
         # TODO: 0/1 for goal reached, -1 for draw or not over
      def getFeatures(self, state, player, featureSet='shared'):
         # TODO: path_diff, walls_diff (+ my_progress, opp_progress for 'custom'), scaled to ~[-1,1]
      def render(self, state=None):
         # TODO: text grid with pawns and walls
      # LAST: fill HEURISTIC_WEIGHTS 
