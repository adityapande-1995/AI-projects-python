#! python3 
import numpy as np

class state1:
    def __init__(self, board=None):
        self.valuef = 0 # Minimax value
        self.optimove = 0 # Optimum move
        self.sym = 1 # play from which side ? 1 = computer, -1 = human
        self.depth = 0 # Tree depth
        if not isinstance(board,np.ndarray):
            self.board = np.array([[0,0,0],[0,0,0],[0,0,0]])
        else :
            self.board = np.copy(board)

    def whoWon(self):
        # Check columns and rows:
        temp1 = np.sum(self.board, axis=0); temp2 = np.sum(self.board, axis=1) 
        if np.amax(temp1) == 3 or np.amax(temp2) == 3:
            return 1
        if np.amin(temp1) == -3 or np.amax(temp2) == -3:
            return -1
        
        # Check diagonals
        if np.trace(self.board) == 3 or np.trace(np.flip(self.board, axis=1)) == 3:
            return 1
        if np.trace(self.board) == -3 or np.trace(np.flip(self.board, axis=1)) == -3:
            return -1
        
        # Nothing works out, nobody won (draw or nonterminal state)
        return 0
    
    def isTerminal(self):
        return not (0 in self.board)   
    
    def play(self, pos, symbol):
        i,j = pos
        if self.board[i,j] == 0:
            temp = np.copy(self.board)
            temp[i,j] = symbol
            return state1(board=temp)
        else:
            print("Invalid move")
    
    def show(self):
        #print(self.board)
        temp = np.array([['-','-','-'],['-','-','-'],['-','-','-']])
        for i in range(3):
            for j in range(3):
                if self.board[i,j] == 1:
                    temp[i,j] = 'o' # Agent has o
                if self.board[i,j] == -1:
                    temp[i,j] = 'x' # Human has x
        
        space = "\t"
        for i in range(self.depth): 
            space += "\t"
        rep = "\n"+space
        print(space + str(temp).replace('\n',rep))
    
    def build_tree(self):
        self.subtree = {}
        if self.whoWon() == 0: # Win states won't run
            for i in range(3):
                for j in range(3):
                    if self.board[i,j] == 0: # Draws won't run
                        temp = self.play((i,j), self.sym)
                        temp.sym = -1*self.sym  # Invert move 
                        temp.depth = self.depth + 1 # State knows its depth 
                        temp.build_tree()
                        self.subtree[(i,j)] = temp

    def show_tree(self, depth=2):
        space = "\t"
        for i in range(self.depth): space += "\t"

        self.show(), print(space,"Node depth : ",self.depth, " Value : ",self.valuef, " Opti Move : ",self.optimove)
        if (depth > 0):
            for key, val in self.subtree.items():
                val.show_tree(depth-1)

    def set_optimum_move_and_val(self):
        if (not self.isTerminal()) and (self.whoWon() == 0): # Further game tree exists, game is still playable
            if self.sym == 1: # MAX behaviour
                maxval = -500
                for loc, s in self.subtree.items():
                    s.set_optimum_move_and_val()
                    if s.valuef > maxval:
                        maxval = s.valuef
                        maxvalL = loc

                self.valuef = maxval; self.optimove = maxvalL
        
            else: # MIN behaviour
                minval = 500
                for loc, s in self.subtree.items():
                    s.set_optimum_move_and_val()
                    if s.valuef < minval:
                        minval = s.valuef
                        minvalL = loc

                self.valuef = minval; self.optimove = minvalL

        if self.isTerminal() and (self.whoWon == 0): # Draw state
            self.valuef = 0 ; self.optimove = None
        
        if self.whoWon() == 1: # Computer won, dosen't matter if board full or not
            self.valuef = 100 ; self.optimove = None
           
        if self.whoWon() == -1: # Human won, dosen't matter if board full or not
            self.valuef = -100 ; self.optimove = None
            

class Human:
    def __init__(self):
        self.symbol = -1

    def play(self,s):
        i,j = input("Enter row,col :").split(",")
        pos = [int(i),int(j)]
        newState = s.play(pos,self.symbol)
        return newState

class IntelliAgent:
    def __init__(self):
        self.symbol = 1
    
    def play(self, s):
        
        # Think
        print("Thinking..")
        print("Generating tree..")
        s.build_tree()
        print("Computing optimal move..")
        s.set_optimum_move_and_val()
        # s.show_tree(3) 
        pos = s.optimove
        newState = s.play(pos,self.symbol)
        return newState

class RandomAgent:
    def __init__(self):
        self.symbol = 1
    
    def play(self, s):
        t1 = []
        for i in range(3):
            for j in range(3):
                if s.board[i,j] == 0:
                    t1.append([i,j])

        index = np.random.randint(len(t1), size=1)[0]
        pos = t1[index]
        newState = s.play(pos,self.symbol)
        return newState



        