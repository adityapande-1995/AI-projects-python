import numpy as np
from bird import *
import copy,random

def new_generation(info,saved_birds = None):
    if not saved_birds: # First generation
        return [Bird(info) for i in range(0,info["popcount"])]    
    else:
        # Make new generation
        # Calculate fitness
        sum = 0
        for b in saved_birds:
            sum += b.score
        for b in saved_birds:
            b.fitness = (b.score/sum)**3

        for b in saved_birds: # Prev gen scores 
            print("Score, Fitness, brain signature : ",b.score, b.fitness, b.brain.signature())
        best_score = copy.deepcopy(saved_birds[-1].score)
        # Selection of best ones
        newB = []
        for i in range(0,info["popcount"]):
            newB.append(pickone(info,saved_birds))

        return newB,best_score

def pickone(info,s):
    method = 2
    if method == 0: # Shiffman's algo
        #fitlist = [Lo.fitness for Lo in s]
        #M,m = max(fitlist), min(fitlist)
        index = 0
        r = np.random.random()
        while r > 0:
            r = r -  s[index].fitness #map(s[index].fitness,m ,M ,0,1)
            index += 1
    
        index -= 1
        child = Bird(info, brain= s[index].brain )
        child.mutate(q=0.2)
        return child
    elif method == 1: # Simply select any of top 10
        index = np.random.randint(1,10)
        child = Bird(info, brain= s[-index].brain )
        child.mutate(q=1)
        return child
    elif method == 2: # My algo 
        fitlist = [np.exp(Lo.score/100) for Lo in s]
        M,m = max(fitlist), min(fitlist)
        fitlist_norm = [map(item,m,M,0,1) for item in fitlist]
        string = []
        for index in range(0,len(fitlist)):
            string += [str(index)]*int(fitlist_norm[index]*100)
        
        select = random.choice(string)
        child = Bird(info, brain= s[int(select)].brain )
        child.mutate(q=0.7)
        return child

          



def map(n, start1, stop1, start2, stop2):
    return ((n-start1)/(stop1-start1))*(stop2-start2)+start2