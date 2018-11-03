#!usr/bin/python3
import pygame
import numpy as np
from bird import *
from pipe import *

# Control parameters
info = {"width":400, "height":600 }

# Pygame setup
pygame.init()
info["screen"] = pygame.display.set_mode([ info["width"], info["height"] ])
pygame.display.set_caption("Flappy bird manual mode")
done = False
clock = pygame.time.Clock()

# Go
framecount = 0
b = Bird(info)
pipes = []
pipes.append(Pipe(info))

while not done:
    clock.tick(100)

    # Event handler
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif (event.type == pygame.KEYDOWN) and (event.key == pygame.K_SPACE) :
            b.up()
                              
    info["screen"].fill((75,75,75))

    # Do stuff
    framecount += 1
    if (framecount % 100 == 0): pipes.append(Pipe(info))
    b.update()
    b.show()
    for p in pipes:
        p.show()
        p.update()
        if p.hits(b): print("hit")
        if p.offscreen(): pipes.pop(pipes.index(p)) 


    # Update drawing
    pygame.display.flip()

pygame.quit()


