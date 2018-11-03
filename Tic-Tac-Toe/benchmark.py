#! python3
from m1 import *
import numpy as np

def game():
    # Game start
    p1 = IntelliAgent() # +1
    p2 = RandomAgent(); p2.symbol = -1 # -1
    curr_player = -1

    B = state1()
    print("Board initial stage : \n");B.show()

    while True:
        print("Board state (filled completely ?, whoWon?) :", B.isTerminal(), B.whoWon())

        if B.isTerminal() or B.whoWon() != 0: # if (the board is filled) or (someone has won)
            print("Game over !") ; B.show()
            winner = B.whoWon()
            if winner == p2.symbol:
                print("Random Agent won !")
                return "random"
            elif winner == p1.symbol:
                print("IntelliAgent won ! #AI-ftw !!")
                return "ai"
            elif winner == 0:
                print("Draw")
                return "draw"

        else: # Next move can be played
            if curr_player == p2.symbol: # Random agent plays
                print("Random agent playing..")
                B = p2.play(B)
                B.show()
                curr_player = p1.symbol
            elif curr_player== p1.symbol: # Agent plays
                print("\nIntelliAgent playing..")
                B = p1.play(B)
                B.show()
                curr_player = p2.symbol

       
def main():
    TOTAL = 50
    won = 0
    drawn = 0
    lost = 0

    for i in range(TOTAL):
        print("\n *** Round ",i+1 , " ***")
        result = game()
        if result == "random":
            lost += 1
        elif result == "ai":
            won += 1
        elif result == "draw":
            drawn += 1

        print("Result so far (won,lost,drawn) : ",won,lost,drawn)
    

main()

