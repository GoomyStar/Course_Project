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

         if index != own_store:
            new[14] = 1 - player
         return new
   
   def isTerminal(self, state=None):
      if state is None:
         state = self.state

      side0 = state[0:6].sum()       # total seeds in player 0's pits
      side1 = state[7:13].sum()   # total seeds in player 1's pits

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
      for i in range(0, 5):             # cups 0 to 5
         bottom = bottom + str(int(state[i])) + " "

      text = "P1  " + top + "\n"
      text = text + "[" + str(int(state[13])) + "]            [" + str(int(state[6])) + "]\n"
      text = text + "P0  " + bottom + "\n"
      text = text + "(player to move: " + str(int(state[14])) + ")"
      return text
