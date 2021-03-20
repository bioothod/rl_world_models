from __future__ import annotations
from typing import *

import argparse
import os
import PIL
import pygame

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--map_image', type=str, required=True, help='Map image')
parser.add_argument('--num_cars', type=int, default=1, help='Number of cars on the map')
parser.add_argument('--output_dir', type=str, help='When set, save rendered frames there')
FLAGS = parser.parse_args()

class Coord:
    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)

    def __add__(self, c: Coord) -> Coord:
        x = self.x + c.x
        y = self.y + c.y
        return Coord(x, y)

    def __sub__(self, c: Coord) -> Coord:
        x = self.x - c.x
        y = self.y - c.y
        return Coord(x, y)

    def __mult__(self, f: float) -> Coord:
        x = self.x * f
        y = self.y * f
        return Coord(x, y)

    def __truediv__(self, f: float) -> Coord:
        x = self.x / f
        y = self.y / f
        return Coord(x, y)

    def __repr__(self) -> str:
        return f'({self.x:d}.{self.y:d})'

    def export(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def export_int(self) -> Tuple[int, int]:
        return (int(self.x), int(self.y))

    def rotate(self, angle: float) -> Coord:
        c = np.cos(angle)
        s = np.sin(angle)

        xr = self.x*c - self.y*s
        yr = self.x*s + self.y*c

        return Coord(xr, yr)

class Polygon:
    def __init__(self, coords: Sequence[Coord]):
        self.coords = coords

    def __add__(self, d: Coord) -> Polygon:
        return Polygon([c+d for c in self.coords])

    def __sub__(self, d: Coord) -> Polygon:
        return Polygon([c-d for c in self.coords])

    def rotate(self, angle: float) -> Polygon:
        return Polygon([c.rotate(angle) for c in self.coords])

    def export_int(self) -> Sequence[Tuple[int, int]]:
        return [c.export_int() for c in self.coords]

    def __getitem__(self, idx: int) -> Coord:
        if idx >= len(self.coords):
            raise IndexError(f'polygon has {len(self.coords)} coords, requested index {idx} is out of range')

        return self.coords[idx]

class Beam:
    def __init__(self, config: Config, c0: Coord, c1: Coord):
        self.config = config
        self.start = c0
        self.go_right = c1.x >= c0.x

        t = c1 - c0

        self.a = float(t.y) / (float(t.x) + 1e-10)
        self.b = c0.y - c0.x * self.a

    def run(self, x: float) -> float:
        return self.a * x + self.b

    def coord_is_ok(self, x, y) -> bool:
        return x < self.config.image_width and y < self.config.image_height and x >= 0 and y >= 0

    def intersect(self) -> Coord:
        x = self.start.x
        y = self.run(x)

        step = 1
        while self.coord_is_ok(x, y):
            p = self.config.image_map.getpixel((x, y))
            if p == self.config.nonroad_colour:
                break

            if self.go_right:
                nx = x + step
                if nx >= self.config.image_width:
                    break
            else:
                nx = x - step
                if nx < 0:
                    break

            x = nx
            y = self.run(x)
            step *= self.config.step_multiplier

        return Coord(x, y)

class Car:
    coords: Polygon

    def __init__(self, config: Config, center: Coord, angle: float, window: Optional[pygame.Surface]) -> None:
        self.config = config
        self.center = center
        self.angle = angle
        self.window = window
        self.endpoints = []
        self.is_dead = False

        self.velocity = 0.
        self.steering_angle = 0.

        self.update_coords()

    def crash(self) -> None:
        self.is_dead = True

    def get_polygon(self) -> Polygon:
        return self.coords

    def update_coords(self) -> None:
        x0 = self.center.x - self.config.car_length / 2
        x1 = self.center.x + self.config.car_length / 2
        y0 = self.center.y - self.config.car_width / 2
        y1 = self.center.y + self.config.car_width / 2

        c0 = Coord(x0, y0)
        c1 = Coord(x1, y0)
        c2 = Coord(x1, y1)
        c3 = Coord(x0, y1)

        self.coords = Polygon([c0, c1, c2, c3])
        self.rotate(self.angle)
        self.run_beams()

    def rotate(self, angle) -> None:
        self.coords -= self.center # pytype: disable=attribute-error
        self.coords = self.coords.rotate(angle)
        self.coords += self.center

    def run_beams(self) -> None:
        c0 = self.coords[0]
        c1 = self.coords[1]
        c2 = self.coords[2]
        c3 = self.coords[3]

        beams_left = []
        beams_right = []

        beam_points = [c0, c1, c2, c3, (c0+c1)/2, (c1+c2)/2, (c2+c3)/2, (c3+c0)/2]

        self.endpoints = []

        for b in beam_points:
            beam = Beam(self.config, self.center, b)
            endpoint = beam.intersect()

            self.endpoints.append(endpoint)

    def step(self, acceleration_value, rotation_value) -> None:
        if self.is_dead:
            return

        rotation_value = np.clip(rotation_value, self.config.rotation_value_min, self.config.rotation_value_max)
        acceleration_value = np.clip(acceleration_value, self.config.acceleration_value_min, self.config.acceleration_value_max)

        self.steering_angle += rotation_value
        self.steering_angle = np.clip(rotation_value, self.config.min_steering_angle, self.config.max_steering_angle)

        if acceleration_value < 0:
            acceleration_value = acceleration_value * self.config.backward_acceleration_ratio

        self.angle += self.steering_angle
        self.velocity += acceleration_value

        vel_x = self.velocity * np.cos(self.angle)
        vel_y = self.velocity * np.sin(self.angle)

        step = 1

        off = Coord(vel_x*step, vel_y*step)

        self.center += off
        self.update_coords()
        self.run_beams()

    def render(self) -> None:
        colour = self.config.car_colour
        if self.is_dead:
            colour = self.config.dead_car_colour

        pygame.draw.polygon(self.window, colour, self.coords.export_int())

        #if not self.is_dead:
        #    for ep in self.endpoints:
        #        pygame.draw.line(self.window, self.config.beam_colour, self.center.export_int(), ep.export_int())

class Config:
    image_path = ''
    image_map = None
    image_width = None
    image_height = None

    car_width = 3
    car_length = 8

    step_multiplier = 1.3

    start_colour = (0xff, 0xee, 0x00)
    nonroad_colour = (255, 255, 255)
    car_colour = (255, 65, 0)
    beam_colour = (175, 65, 255)
    dead_car_colour = (255, 0, 0)

    backward_acceleration_ratio = 0.5

    max_steering_angle = np.pi/4
    min_steering_angle = -np.pi/4

    rotation_value_abs = 10. / 180. * np.pi
    rotation_value_min = -rotation_value_abs
    rotation_value_max = rotation_value_abs

    acceleration_value_min = -0.1
    acceleration_value_max = 0.2

class MapGame:
    window: pygame.Surface
    bg_surf: pygame.Surface
    clock: pygame.time.Clock

    def __init__(self, config: Config):
        self.config = config

        self.cars = []

        self.image = PIL.Image.open(self.config.image_path)
        config.image_map = self.image

        bbox = self.image.getbbox()
        self.config.image_width = bbox[2]
        self.config.image_height = bbox[3]

    def locate_start(self) -> Tuple[Coord, Coord]:
        a = np.asarray(self.image)
        start = np.nonzero((a[:, :, 0] == self.config.start_colour[0]) & (a[:, :, 1] == self.config.start_colour[1]) & (a[:, :, 2] == self.config.start_colour[2]))
        min_x = start[1].min()
        min_y = start[0].min()
        max_x = start[1].max()
        max_y = start[0].max()

        min_c = Coord(min_x, min_y)
        max_c = Coord(max_x, max_y)
        return (min_c, max_c)

    def add_car(self, coord: Coord) -> None:
        angle = -90
        angle = angle * np.pi / 180.
        c = Car(self.config, coord, angle, self.window)
        self.cars.append(c)

    def init_render(self) -> None:
        pygame.init()

        self.window = pygame.display.set_mode(size=(self.config.image_width, self.config.image_height))

        self.bg_surf = pygame.image.load(self.config.image_path).convert()
        self.clock = pygame.time.Clock()

    def render(self) -> None:
        self.window.blit(self.bg_surf, (0, 0))
        for car in self.cars:
            car.render()

        pygame.display.update()

    def step(self) -> None:
        for car in self.cars:
            car.step(acceleration_value=0.01, rotation_value=0.001)

        for car in self.cars:
            coords = car.get_polygon()
            for c in coords:
                p = self.config.image_map.getpixel((c.x, c.y))
                if p == self.config.nonroad_colour:
                    car.crash()
                    break

def main() -> Any:
    config = Config()
    config.image_path = FLAGS.map_image

    map_game = MapGame(config)
    map_game.init_render()

    min_c, max_c = map_game.locate_start()
    diff_x = max_c.x - min_c.x
    diff_y = max_c.y - min_c.y

    num_cars = 3
    step_x = diff_x / num_cars
    step_y = diff_y / num_cars
    for i in range(num_cars):
        c = min_c + Coord((step_x + config.car_width)*i, step_y*i)
        map_game.add_car(c)

    run = True
    step = 0
    if FLAGS.output_dir:
        os.makedirs(FLAGS.output_dir, exist_ok=True)

    while run:
        # set game speed to 30 fps
        map_game.clock.tick(30)
        # ─── CONTROLS ───────────────────────────────────────────────────────────────────
        # end while-loop when window is closed
        get_event = pygame.event.get()
        for event in get_event:
            if event.type == pygame.QUIT:
                run = False
        # get pressed keys, generate action
        get_pressed = pygame.key.get_pressed()

        keys = np.array(get_pressed)
        pressed_keys = np.where(keys != 0)
        #print(pressed_keys)
        #action = pressed_to_action(pressed_keys)
        # calculate one step
        #environment.step(action)
        # render current state

        map_game.step()
        map_game.render()

        if FLAGS.output_dir:
            pygame.image.save(map_game.window, os.path.join(FLAGS.output_dir, f'/stream{step:04d}.png'))
        step += 1
    pygame.quit()

if __name__ == '__main__':
    main()
