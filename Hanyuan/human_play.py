import argparse
import os
import shutil
import numpy as np
import torch
import pygame
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE, K_SPACE, K_UP

from src.flappy_bird import FlappyBird
from src.utils import pre_processing

current_experiment_name = 'original name with pip gap 110'
player_name = 'Hanyuan'

device = "cuda" if torch.cuda.is_available() \
                      else "mps" if torch.backends.mps.is_available() \
                      else "cpu"
print("Playing on {}".format(device))

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird""")
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args

# def get_player_name():
#     pygame.init()
#     screen = pygame.display.set_mode((480, 360))
#     font = pygame.font.Font(None, 32)
#     clock = pygame.time.Clock()
#     input_box = pygame.Rect(100, 100, 140, 32)
#     color_inactive = pygame.Color('lightskyblue3')
#     color_active = pygame.Color('dodgerblue2')
#     color = color_inactive
#     active = False
#     text = ''
#     done = False

#     while not done:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 done = True
#             if event.type == pygame.MOUSEBUTTONDOWN:
#                 if input_box.collide_point(event.pos):
#                     active = not active
#                 else:
#                     active = False
#                 color = color_active if active else color_inactive
#             if event.type == pygame.KEYDOWN:
#                 if active:
#                     if event.key == pygame.K_RETURN:
#                         done = True
#                     elif event.key == pygame.K_BACKSPACE:
#                         text = text[:-1]
#                     else:
#                         text += event.unicode

#         screen.fill((30, 30, 30))
#         txt_surface = font.render(text, True, color)
#         width = max(200, txt_surface.get_width()+10)
#         input_box.w = width
#         screen.blit(txt_surface, (input_box.x+5, input_box.y+5))
#         pygame.draw.rect(screen, color, input_box, 2)

#         pygame.display.flip()
#         clock.tick(30)

#     return text


def play(opt):
    game_state = FlappyBird()
    image, reward, terminal = game_state.next_frame(0)
    image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)
    image = torch.from_numpy(image).to(device)
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]
    state = state.to(device)
    
    # play 5 rounds, this settings help players to get familiar with the game
    # and give them time to think about the strategy after each 5 rounds
    total_rounds = 3
    rounds = 1
    scores = [None] * total_rounds

    highest_score_this_round = 0

    while rounds <= total_rounds:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                quit()
            elif event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                action = 1
            else:
                action = 0

        next_image, reward, terminal = game_state.next_frame(action)

        if terminal:
            print('Game over! Score: {}'.format(highest_score_this_round))
            scores[rounds - 1] = highest_score_this_round
            rounds += 1

            # reset the game
            highest_score_this_round = 0
            game_state = FlappyBird()
            image, reward, terminal = game_state.next_frame(0)
            image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)
            image = torch.from_numpy(image).to(device)
            state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]
            state = state.to(device)

        else:
            next_image = pre_processing(next_image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)
            next_image = torch.from_numpy(next_image).to(device)
            state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]
            # print('current score: {}'.format(game_state.score))
            highest_score_this_round = max(highest_score_this_round, game_state.score)
            # print('highest score up to now: {}'.format(highest_score_this_round))
    
    # After 20 rounds, write the scores to a txt file
    with open('score_record/' + current_experiment_name + 'human scores.txt', 'w') as f:
        f.write(f"Player: {player_name}\n")
        for score in scores:
            f.write(f"{score}\n")



if __name__ == "__main__":
    opt = get_args()
    play(opt)