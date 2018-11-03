import numpy as np
import pygame
from nn import *

class Bird:
    def __init__(self,info, brain = None):
        self.info = info
        self.y = self.info["height"]/2
        self.x = 25
        self.gravity = 0.6
        self.lift = -15
        self.velocity = 0
        self.score = 0
        self.fitness = 0
        if brain:
            self.brain = brain.deepcopy()
        else:
            self.brain = NeuralNetwork_1([5,4,1])
    
    def show(self):
        pygame.draw.circle( self.info["screen"], (0,255,0), np.array([self.x, int(self.y)]), 10)
    
    def update(self):
        self.score += 1
        self.velocity += self.gravity
        self.velocity = np.clip(self.velocity, -10,10)
        self.y += self.velocity

        if (self.y > self.info["height"]):
            self.y = self.info["height"]
            self.velocity = 0
        if (self.y < 0):
            self.y = 0
            self.velocity = 0
    
    def up(self):
        self.velocity += self.lift

    def think(self,pipes):
        # Find out the closest pipe
        closest = None
        closestD = 10000000
        for P in pipes:
            d = P.x + P.w - self.x
            if d < closestD and d > 0:
                closest = P
                closestD = d

        # Generate input features and predict using brain
        input = [0,0,0,0,0]
        input[0] = self.y/self.info["height"]
        input[1] = closest.top/self.info["height"]
        input[2] = closest.bottom/self.info["height"]
        input[3] = closest.x/self.info["width"]
        input[4] = self.velocity/10

        output = self.brain.predict(input)
        #print("Input ",input)
        #print("Output ", output[0])
        if output[0] > 0.5:
            self.up()

    def mutate(self,q):
        self.brain.mutate(q)



        






