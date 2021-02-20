import argparse
import logging
import os
import pickle

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

logger = logging.getLogger('plain_world')

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', required=True, type=str, help='Train directory')
parser.add_argument('--input_size', type=int, default=16, help='Input world size')
FLAGS = parser.parse_args()

class World:
    def __init__(self, input_size, num_objects):
        shape = [input_size, input_size]
        world = np.random.choice(np.arange(num_objects + 1), shape)

        self.pos = np.random.rand(2) * FLAGS.input_size
        self.pos = self.pos.astype(int)

        self.num_oh = num_objects + 1
        shape_oh = [input_size, input_size, self.num_oh]
        world_oh = np.zeros(shape_oh)
        world_oh[:, :, 0] = np.where(world == 0, 1, 0)
        for i in range(1, self.num_oh):
            world_oh[:, :, i] = np.where(world == i, 1, 0)

        self.world = world
        self.world_oh = world_oh

    def draw(self):
        fig, axs = plt.subplots(1, self.num_oh+1)
        axs[0].imshow(self.world)

        for i in range(self.num_oh):
            axs[i+1].imshow(self.world_oh[:, :, i])

        for i in range(self.num_oh+1):
            # (left, bottom), width, height
            rect = Rectangle((self.pos[1]-0.5, self.pos[0]-0.5), 1, 1)
            pc = PatchCollection([rect], facecolor='r', alpha=0.9, edgecolor='None')

            axs[i].add_collection(pc)

        plt.show()

def main():
    #tf.random.set_seed(0)
    #np.random.seed(0)

    os.makedirs(FLAGS.train_dir, exist_ok=True)

    logger.propagate = False
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    logger.addHandler(handler)

    handler = logging.FileHandler(os.path.join(FLAGS.train_dir, 'logfile.log'), 'a')
    handler.setFormatter(fmt)
    logger.addHandler(handler)

    num_objects = 2
    w = World(input_size=FLAGS.input_size, num_objects=num_objects)

    w.draw()




if __name__ == '__main__':
    main()
