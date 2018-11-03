# Tic tac toe solver

## Introduction
A tic tac toe game solving agent using minimax algorithm. THe game contains 2 agents :
1. The IntelliAgent : Runs minimax algorithm to figure out moves.
2. The RandomAgent : Randomly plays in any empty spot.

## Running
For a human vs agent game, run :
```
python3 play.py
```

To compare the performance of 2 agents by making them play against each other, run:
```
python3 benchmark.py
```
When I ran the code for 50 games, IntelliAgent won 46 of them, drew 4, and lost 0. 

## Further improvements
The policies can be made more efficient using alpha-beta pruning.


