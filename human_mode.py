import sys, argparse
import pygame

sys.path.append('game/')
import flappy_wrapped as game
import cv2
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.optim as optim

KERNEL = np.array([[-1,-1,-1], [-1, 9,-1],[-1,-1,-1]])
SKIP_FRAME = 2
    
if __name__=='__main__':
    
    env = game.GameState()
    
    input("Please Press Space to Start")
    total_rewards = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
         
            # checking if keydown event happened or not
            if event.type == pygame.KEYDOWN:
            
                # if keydown event happened
                # than printing a string to output
                print("A key has been pressed")
                action = int(1)
            else:
                action = int(0)
        
        frame,reward,done = env.frame_step(action)
        total_rewards += reward
        
        for _ in range(SKIP_FRAME):
            frame,reward,done =  env.frame_step(action)
            total_rewards += reward
            if done:
                break

                
        
        
        