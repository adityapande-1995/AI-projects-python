#!usr/bin/python3
import pygame, time
import numpy as np
from bird import *
from pipe import *
import ga
import matplotlib.pyplot as plt

# Control parameters
info = {"width":400, "height":600 ,"popcount":500 ,"generation_count":0, "speedup_cycles":3 }

# Pygame setup
pygame.init() ; pygame.font.init() ; myfont = pygame.font.SysFont('Comic Sans MS', 20)
info["screen"] = pygame.display.set_mode([ info["width"], info["height"] ])
pygame.display.set_caption("Flappy bird Neuroevolution")
done = False
clock = pygame.time.Clock() ; a = time.time()

# Go
framecount = 0; best_score = 0
birds = ga.new_generation(info); savedBirds = []
pipes = []; pipes.append(Pipe(info))
x_plot = [0] ; y_plot = [0] ; plt.ion();plt.title("Top scores per generation") # Plotting initialize
plt.xlabel("Generation") ; plt.ylabel("Score") 


while not done:
    for CYCLES in range(0,info["speedup_cycles"]):
        clock.tick(500)

        # Event handler
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif (event.type == pygame.KEYDOWN) and (event.key == pygame.K_UP) :
                info["speedup_cycles"] += 1
            elif (event.type == pygame.KEYDOWN) and (event.key == pygame.K_DOWN) and info["speedup_cycles"] > 3  :
                info["speedup_cycles"] -= 1    
                              
        info["screen"].fill((75,75,75))

        # Do stuff
        framecount += 1
        if (framecount % 200 == 0): pipes.append(Pipe(info)) # Add pipes to screen
        for b in birds: # Bird takes some action
            b.think(pipes)
            b.update()
            #b.show()
        for p in pipes: # Pipe animation
            #p.show()
            p.update()
            for b in birds: # Kill bird if it hits pipe
                if p.hits(b):
                    savedBirds.append(birds.pop(birds.index(b)))

            if p.offscreen(): pipes.pop(pipes.index(p)) 
    
        # New generation, all birds dead
        if len(birds) == 0:
            info["generation_count"] += 1
            print("\nGeneration ",info["generation_count"], " completed\n")
            birds,best_score = ga.new_generation(info,savedBirds)
            savedBirds = []
            pipes = [Pipe(info,special=True)]
            framecount = 0
            # Plotting
            x_plot.append(info["generation_count"]) ; y_plot.append(best_score)
            plt.plot(x_plot, y_plot); plt.show(); plt.pause(0.0001)

    # Update drawing ( outside cycles loop)    
    for b in birds: b.show()
    for p in pipes: p.show()
    text1 = "Generation " + str(info["generation_count"]) 
    text2 = "Frame skip: " + str(info["speedup_cycles"])
    text3 = "Prev gen best: " + str(best_score)
    text4 = "Time elapsed (min): " + str(    (time.time() - a)//60    )
    text5 = "Surviving birds: " + str(info["popcount"] - len(savedBirds)) + "/" + str(info["popcount"])
    info["screen"].blit( myfont.render(text1, False, (255, 165, 0)) ,(info["width"]-170 ,0))
    info["screen"].blit( myfont.render(text2, False, (255, 165, 0)) ,(info["width"]-170 ,15))
    info["screen"].blit( myfont.render(text3, False, (255, 165, 0)) ,(info["width"]-170 ,30))
    info["screen"].blit( myfont.render(text4, False, (255, 165, 0)) ,(info["width"]-170 ,45))
    info["screen"].blit( myfont.render(text5, False, (255, 165, 0)) ,(info["width"]-170 ,60))
    pygame.display.flip()

pygame.quit()


