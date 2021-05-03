import math
import os
import pygame

import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

import numpy as np

import image_map_game as imap

class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, map_image):
        super().__init__()

        self.config = imap.Config()
        self.config.image_path = map_image

        self.map_game = imap.MapGame(self.config)
        self.map_game.init_render()

        initial_state = self.reset()

        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([+1, +1]), dtype=np.float32)
        #self.observation_space = spaces.Box(low=0, high=255, shape=initial_state.shape, dtype=initial_state.dtype)

    def seed(self, seed=None):
        self.np_random, seed=seeding.np_random(seed)
        return [seed]

    def dist_to_goal(self, car):
        return math.sqrt((car.center.x - self.map_game.goal.x)**2 + (car.center.y - self.map_game.goal.y)**2)

    def step(self, actions):
        #print(f'action: {actions}')
        self.map_game.step([actions])

        state = self.map_game.current_state()[0]

        rewards = []
        dones = []

        for car, orig_dist in zip(self.map_game.cars, self.orig_dists):
            done = car.is_dead
            reward = -1.

            dist = self.dist_to_goal(car)

            if not car.is_dead:
                if dist < 3:
                    reward = 1.
                    done = True
                else:
                    reward = 1. - dist / orig_dist - 0.01

            rewards.append(reward)
            dones.append(done)

        return state, rewards[0], dones[0], {}
    
    def reset(self):
        self.map_game.reset()
        self.map_game.add_cars(1)

        self.orig_dists = [self.dist_to_goal(car) for car in self.map_game.cars]

        initial_state = self.map_game.current_state()[0]

        return initial_state

    def render(self, episode, step, mode='human', close=False):
        self.map_game.render()

        if self.config.output_dir:
            out_dir = os.path.join(self.config.output_dir, f'{episode:06d}')
            os.makedirs(out_dir, exist_ok=True)
            out_fn = os.path.join(out_dir, f'stream{step:04d}.png')
            pygame.image.save(self.map_game.window, out_fn)
