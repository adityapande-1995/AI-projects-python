import numpy as np
import pygame

class Pipe:
    def __init__(self,info,special=None):
        self.info = info
        self.spacing = 150
        self.top = np.random.uniform(0.1*self.info["height"],0.7*self.info["height"])  #np.random.random()*self.info["height"]/2
        self.bottom = self.info["height"] - (self.top + self.spacing) #np.random.random()*self.info["height"]/2
        self.x = self.info["width"]
        self.w = 40
        self.speed = 1
        self.highlight = False
        if self.info["generation_count"] % 2 == 0:
            self.colour = (255,255,255)
        else:
            self.colour = (200,200,200)
        if special:
            self.colour = (128,128,128)

    def show(self):
        if self.highlight:
            colour = (255,0,0)
        else:
            colour = self.colour

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




        