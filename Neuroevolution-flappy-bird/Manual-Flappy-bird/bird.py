import numpy as np
import pygame

class Bird:
    def __init__(self,info):
        self.info = info
        self.y = self.info["height"]/2
        self.x = 25
        self.gravity = 0.6
        self.lift = -15
        self.velocity = 0
    
    def show(self):
        pygame.draw.circle( self.info["screen"], (0,255,0), np.array([self.x, int(self.y)]), 10)
    
    def update(self):
        self.velocity += self.gravity
        self.velocity *= 0.9
        self.y += self.velocity

        if (self.y > self.info["height"]):
            self.y = self.info["height"]
            self.velocity = 0
        if (self.y < 0):
            self.y = 0
            self.velocity = 0
    
    def up(self):
        self.velocity += self.lift


        






