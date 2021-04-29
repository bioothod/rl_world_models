import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

import numpy as np

import image_map_env as imap

class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, map_image):
        super().__init__()

        self.config = imap.Config()
        self.config.image_path = map_image

        self.map_game = imap.MapGame(config)
        self.map_game.init_render()

        initial_state = self.reset()

        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([+1, +1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=initial_state.shape, dtype=initial_state.dtype)

    def seed(self, seed=None):
        self.np_random, seed=seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.imap.step(action)
        done = False
        if self.car[0].is_dead:
            done = True

        state = self.map_game.current_state()

        reward = (self.car[0].center.x - 500)**2 + 

        return state, reward, done, {}
    
    def reset(self):
        self.map_game.reset()
        self.map_game.add_cars(1)

        initial_state = self.map_game.current_state()

        return initial_state

    def render(self, mode='human', close=False):
        self.map_game.render()
