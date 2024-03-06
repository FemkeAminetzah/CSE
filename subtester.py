import os
import sys
import pygame
import numpy as np

sys.path.append("ContinuousSubmarine_agents")
sys.path.append("ContinuousSubmarineEnvironment")
from ple.games.contsubmarinegame import ContSubmarineGame
from run_ple_utils import make_ple_env
import gym

class subtester():
    def __init__(self):
        self.env = gym.make('ContSubmarineGame-v0')
        self.run_game()

    def run_game(self):
        state = self.env.reset()
        clock = pygame.time.Clock()  # Create a clock object to control the frame rate
        fps = 30  # Set the desired frames per second

        for _ in range(100000):
            for event in pygame.event.get():  # Process events to avoid overflow
                if event.type == pygame.QUIT:
                    return

            action = self.env.action_space.sample()
            state, reward, done, _ = self.env.step(action)
            self.env.render()

            if done:
                state = self.env.reset()

            clock.tick(fps)  # Limit the frame rate

        self.env.close()

if __name__ == '__main__':
    subtester()
