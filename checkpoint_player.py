#from __future__ import annotations
from typing import *

import argparse
import logging
import os
import PIL
#import PIL.Image
import pygame
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import image_map_game as imap
import sac

logger = logging.getLogger('player')

def main() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, help='Checkpoint dir')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint prefix')
    parser.add_argument('--map_image', type=str, required=True, help='Map image')
    parser.add_argument('--num_cars', type=int, default=1, help='Number of cars on the map')
    parser.add_argument('--output_dir', type=str, help='When set, save rendered frames there')
    parser.add_argument('--goal', type=str, help='Set goal to these coordinates, format: x.y')
    FLAGS = parser.parse_args()

    logger.propagate = False
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')

    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    logger.addHandler(handler)

    if not FLAGS.checkpoint and not FLAGS.checkpoint_dir:
        logger.info(f'No checkpoints specified, using random initialization')
        restore_path = None
    else:
        restore_path = FLAGS.checkpoint
        if FLAGS.checkpoint_dir:
            restore_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    config = imap.Config()
    config.image_path = FLAGS.map_image

    if FLAGS.goal:
        spl = FLAGS.goal.split('.')
        if len(spl) != 2:
            logger.error(f'Invalid goal {FLAGS.goal}, must be something like 10.20')
        else:
            gx = int(spl[0])
            gy = int(spl[1])
            config.goal = imap.Coord(gx, gy)

            logger.info(f'Setting goal to {config.goal}')

    map_game = imap.MapGame(config)
    map_game.init_render()
    map_game.add_cars(FLAGS.num_cars)

    if FLAGS.output_dir:
        os.makedirs(FLAGS.output_dir, exist_ok=True)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    num_actions = 2
    actor_hidden_layers = [512, 512, 512, 512]
    policy = sac.Actor(actor_hidden_layers, num_actions, name='policy')

    if restore_path:
        checkpoint = tf.train.Checkpoint(policy=policy)
        status = checkpoint.restore(restore_path).expect_partial()
        logger.info(f'Restored from checkpoint {restore_path}')

    done = False

    step = 0
    while not done:
        # set game speed to 30 fps
        map_game.clock.tick(30)

        # end while-loop when window is closed
        get_event = pygame.event.get()
        for event in get_event:
            if event.type == pygame.QUIT:
                break
        # get pressed keys, generate action
        get_pressed = pygame.key.get_pressed()

        keys = np.array(get_pressed)
        pressed_keys = np.where(keys != 0)[0]
        if 20 in pressed_keys:
            break
        if 41 in pressed_keys:
            break

        actions = []
        for i in range(FLAGS.num_cars):
            state = map_game.current_state(i)[0]
            action = policy.get_action(state)
            actions.append(action)

        map_game.step(actions)
        map_game.render()

        if FLAGS.output_dir:
            pygame.image.save(map_game.window, os.path.join(FLAGS.output_dir, f'stream{step:04d}.png'))

        step += 1

    pygame.quit()

if __name__ == '__main__':
    main()
