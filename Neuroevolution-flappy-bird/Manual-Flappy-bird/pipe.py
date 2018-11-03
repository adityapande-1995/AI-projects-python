import numpy as np
import pygame

class Pipe:
    def __init__(self,info):
        self.info = info
        self.top = np.random.random()*self.info["height"]/2
        self.bottom = np.random.random()*self.info["height"]/2
        self.x = self.info["width"]
        self.w = 20
        self.speed = 1
        self.highlight = False

    def show(self):
        if self.highlight:
            colour = (255,0,0)
        else:
            colour = (255,255,255)

        pygame.draw.rect(self.info["screen"], colour, [self.x,0,self.w,self.top])
        pygame.draw.rect(self.info["screen"], colour, [self.x,self.info["height"]-self.bottom,self.w,self.bottom])

    def update(self):
        self.x -= self.speed
    
    def offscreen(self):
        return self.x < -self.w
    
    def hits(self,bird):
        if (bird.y < self.top) or (bird.y > self.info["height"] - self.bottom ):
            if (bird.x > self.x) and (bird.x < self.x + self.w):
                self.highlight = True
                return True
            else:
                self.highlight = False
                return False
        else:
            self.highlight = False
            return False




        