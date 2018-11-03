#!python3
from m1 import *

# Main loop
p1 = IntelliAgent() # +1
p2 = Human()
curr_player = -1

B = state1()
B.show()

while True:
    
    if B.isTerminal() or B.whoWon() != 0: # Check whether game has ended aleady
        print("Game over !")
        winner = B.whoWon()
        if winner == p2.symbol:
            print("Human won !")
        elif winner == p1.symbol:
            print("Agent won ! #AI-ftw !!")
        else:
            print("Draw")
        break
    else: # Next move can be played
        if curr_player == p2.symbol: # Human plays
            B = p2.play(B)
            B.show()
            curr_player = p1.symbol
        elif curr_player== p1.symbol: # Agent plays
            print("\nAgent playing..")
            B = p1.play(B)
            B.show()
            curr_player = p2.symbol
    
    B.show()


