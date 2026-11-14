import problems as prb
import algorithms as alg

'''
This file will contain the code needed to run a demonstration of your project. 

For the progress report, it should contain code that shows a brief demonstration of random agents attempting to solve each problem. 

For the final report, it should contain code that shows a brief demonstration of each algorithm attempting to solve each problem. 

In both cases, to make sure we can grade these all in time, please ensure that each demonstration is just long enough to show some
meaningful behavior. If your problem is chess playing, for example, you might show 5 moves from each player, where one player uses one 
algorithm and one player uses another. If your problem is snake, you might show game long enough for the player to eat two apples or die
(whichever comes first).
'''
# TicTacToe Given demo
demoGame = prb.Game(prb.TicTacToe(),alg.RandomAgent(),alg.RandomAgent())
demoGame.playGame()


# Connect Four (X = player 0, O = player 1): a random game is short, so show all of it
print("\n=== Connect Four: Random vs Random ===")
demoGame = prb.Game(prb.ConnectFour(),alg.RandomAgent(),alg.RandomAgent())
demoGame.playGame()

# Mancala: show the first 20 moves (extra turns appear as one player moving twice in a row)
print("\n=== Mancala: Random vs Random (first 20 moves) ===")
demoGame = prb.Game(prb.Mancala(),alg.RandomAgent(),alg.RandomAgent(),maxMoves=20)
demoGame.playGame()

# Wallz (A = player 0, B = player 1): random play rarely finishes, so show the first 24 moves
print("\n=== Wallz: Random vs Random (first 24 moves) ===")
demoGame = prb.Game(prb.Wallz(),alg.RandomAgent(),alg.RandomAgent(),maxMoves=24)
demoGame.playGame()
