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
      'shared': {'store_diff': 10.0},
      'custom': {'store_diff': 10.0},
   }

   def __init__(self):
      self.state = np.full(15, self.SEEDS, dtype=int)   # 15 numbers total for spaces and stores.
      self.state[6] = 0                                 # player 0's store
      self.state[13] = 0                                # player 1's store
      self.state[14] = 0                                # player 0 moves first
      self.ticks = -1

   def getCurrentPlayer(self, state=None):
      if state is None:
         state = self.state
      return int(state[14])

   def getLegalMoves(self, state=None):
      if state is None:
         state = self.state
      base = 7 * int(state[14])          # Initial states of 0 for player 0, and 7 for player 1
      moves = []
      for i in range(self.PITS):
         if state[base + i] > 0:
            moves.append(i)
      return moves

   def getSuccessor(self, move, state):
      new = state.copy()
      player = int(new[14])

      # which store is mine, which is the opponent's?
      if player == 0:
         own_store = 6
         opp_store = 13
      else:
         own_store = 13
         opp_store = 6

      # 1. pick up the seeds from the pit I chose
      index = 7 * player + move
      seeds = new[index]
      new[index] = 0           # the pit is now empty

      # 2. sow them one at a time
      while seeds > 0:
         index = (index + 1) % 14
         if index == opp_store:
            continue
         new[index] = new[index] + 1
         seeds = seeds - 1

      # 3. capture: last seed landed in an empty pit on my side, and the opposite pit has seeds
      my_side_start = 7 * player
      if index >= my_side_start and index <= my_side_start + 5 and new[index] == 1 and new[12 - index] > 0:
         new[own_store] = new[own_store] + new[12 - index] + 1
         new[index] = 0
         new[12 - index] = 0

      # 4. whose turn is next? (landing in my own store means I go again)
      if index != own_store:
         new[14] = 1 - player

      # 5. end of game: if one side is empty, the other side's seeds go to its owner's store
      side0 = new[0:6].sum()
      side1 = new[7:13].sum()
      if side0 == 0 or side1 == 0:
         new[6] = new[6] + side0
         new[13] = new[13] + side1
         for i in range(0, 6):
            new[i] = 0
         for i in range(7, 13):
            new[i] = 0

      return new

   def isTerminal(self, state=None):
      if state is None:
         state = self.state

      side0 = state[0:6].sum()       # total seeds in player 0's pits
      side1 = state[7:13].sum()      # total seeds in player 1's pits

      if side0 == 0 or side1 == 0:
         return True
      return False

   def getWinner(self, state=None):
      if state is None:
         state = self.state

      if state[6] > state[13]:
         return 0
      if state[13] > state[6]:
         return 1
      return -1

   def getFeatures(self, state, player, featureSet='shared'):
      other = 1 - player
      my_store = state[6 + 7 * player]
      their_store = state[6 + 7 * other]

      f = {
         'store_diff': (my_store - their_store) / 24.0,
      }
      return f

   def render(self, state=None):
      if state is None:
         state = self.state

      top = ""
      for i in range(12, 6, -1):          # cups 12, 11, 10, 9, 8, 7
         top = top + str(int(state[i])) + " "

      bottom = ""
      for i in range(0, 6):               # cups 0 to 5
         bottom = bottom + str(int(state[i])) + " "

      text = "P1  " + top + "\n"
      text = text + "[" + str(int(state[13])) + "]            [" + str(int(state[6])) + "]\n"
      text = text + "P0  " + bottom + "\n"
      text = text + "(player to move: " + str(int(state[14])) + ")"
      return text
 
 
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
      self.N = size
      self.W = walls
      self.maxPlies = maxPlies
      self.M = (size - 1) * (size - 1)
      self.state = np.zeros(8 + 2 * self.M, dtype=int)
      self.state[0] = size - 1       # player 0 row: bottom
      self.state[1] = size // 2      # player 0 column: middle
      self.state[2] = 0              # player 1 row: top
      self.state[3] = size // 2      # player 1 column: middle
      self.state[4] = walls
      self.state[5] = walls
      self.ticks = -1

   def getCurrentPlayer(self, state=None):
      if state is None:
         state = self.state
      return int(state[6])

   def _H(self, state):
      return state[8 : 8 + self.M].reshape(self.N - 1, self.N - 1)

   def _V(self, state):
      return state[8 + self.M : 8 + 2 * self.M].reshape(self.N - 1, self.N - 1)

   def _isBlocked(self, state, r1, c1, r2, c2):
      H = self._H(state)
      V = self._V(state)
      N = self.N
      if c1 == c2:                       # moving up or down
         top = min(r1, r2)
         if c1 < N - 1 and H[top][c1 - 1] == 1:
            return True
         if c1 > 0 and H[top][c1] == 1:
            return True
         return False
      else:                              # moving left or right
         left = min(c1, c2)
         if r1 < N - 1 and V[r1][left] == 1:
            return True
         if r1 > 0 and V[r1 - 1][left] == 1:
            return True
         return False
   def _pawnMoves(self, state, player):
      r = int(state[2 * player])
      c = int(state[2 * player + 1])
      moves = []
      directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # up, down, left, right
      for d in directions:
         nr = r + d[0]
         nc = c + d[1]
         # stay on the board
         if nr < 0 or nr >= self.N or nc < ___ or nc >= self.N:
            continue
         # is a wall in the way?
         if self._isBlocked(state, r, c, nr, nc):
            continue
         moves.append((0, nr, nc))
      return moves
      def _shortestPath(self, state, player):
         # TODO: BFS to goal row; steps or None.  CHECK: empty 9x9 -> 8
         pass
      def _wallMoves(self, state, player):
         # TODO: no overlap/crossing; BOTH players must still have a path.  CHECK: opening -> 128
         pass
      def getLegalMoves(self, state=None):
         # TODO: pawn moves + wall moves.  CHECK: opening -> 131
         pass
      def getSuccessor(self, move, state):
         # TODO: copy; apply pawn move or wall (-1 wall left); plies+1; flip player
         pass
      def isTerminal(self, state=None):
         # TODO: a pawn on its goal row, or plies >= maxPlies
         pass
      def getWinner(self, state=None):
         # TODO: 0/1 for goal reached, -1 for draw or not over
         pass
      def getFeatures(self, state, player, featureSet='shared'):
         # TODO: path_diff, walls_diff (+ my_progress, opp_progress for 'custom'), scaled to ~[-1,1]
         pass
      def render(self, state=None):
         # TODO: text grid with pawns and walls
         pass
      # LAST: fill HEURISTIC_WEIGHTS 
