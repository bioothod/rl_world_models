from __future__ import annotations
from typing import *

import argparse
import os
import PIL
#import PIL.Image
import pygame
import sys

import matplotlib.pyplot as plt
import numpy as np

if sys.version_info < (3, 7):
    class Coord:
        pass
    class Polygon:
        pass
    class Config:
        pass


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

    def __mul__(self, f: float) -> Coord:
        x = self.x * f
        y = self.y * f
        return Coord(x, y)
    def __rmul__(self, f: float) -> Coord:
        return self.__mul__(f)

    def __truediv__(self, f: float) -> Coord:
        x = self.x / f
        y = self.y / f
        return Coord(x, y)

    def __repr__(self) -> str:
        return f'({self.x:.2f},{self.y:.2f})'

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

        if np.abs(c1.x - c0.x) < np.abs(c1.y - c0.y):
            self.reverse = True
            self.a = float(c1.x - c0.x) / float(c1.y - c0.y)
            self.b = c0.x - c0.y * self.a

            if c1.y > c0.y:
                self.diff = Coord(0, self.config.lidar_step)
            else:
                self.diff = Coord(0, -self.config.lidar_step)
        else:
            self.reverse = False
            self.a = float(c1.y - c0.y) / float(c1.x - c0.x)
            self.b = c0.y - c0.x * self.a

            if c1.x > c0.x:
                self.diff = Coord(self.config.lidar_step, 0)
            else:
                self.diff = Coord(-self.config.lidar_step, 0)

    def run(self, c: Coord) -> Coord:
        if self.reverse:
            x = self.a * c.y + self.b
            return Coord(x, c.y)
        else:
            y = self.a * c.x + self.b
            return Coord(c.x, y)

    def coord_is_ok(self, c: Coord) -> bool:
        return c.x < self.config.image_width and c.y < self.config.image_height and c.x >= 0 and c.y >= 0

    def intersect(self) -> Coord:
        c = self.start

        while True:
            p = self.config.image_map.getpixel(c.export_int())
            if p in self.config.obstacle_colours:
                break

            nc = c + self.diff
            nc = self.run(nc)
            if not self.coord_is_ok(nc):
                break

            c = nc

        return c

class Car:
    coords: Polygon

    def __init__(self, config: Config, center: Coord, angle: float, window: Optional[pygame.Surface]) -> None:
        self.config = config
        self.center = center
        self.angle = angle
        self.window = window
        self.endpoints = []
        self.is_dead = False

        self.set_goal(config.goal)

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

    def set_goal(self, goal):
        self.goal = goal

    def bound(self, p):
        x = p.x
        y = p.y

        if x > self.config.image_width:
            x = self.config.image_width
        if y > self.config.image_height:
            y = self.config.image_height

        return Coord(x, y)

    def run_beams(self) -> None:
        c0 = self.coords[0]
        c1 = self.coords[1]
        c2 = self.coords[2]
        c3 = self.coords[3]

        beams_left = []
        beams_right = []

        beam_points = [c0, c1, c2, c3, (c0+c1)/2, (c1+c2)/2, (c2+c3)/2, (c3+c0)/2]
        #beam_points = [(c0+c1)/2, (c1+c2)/2, (c2+c3)/2, (c3+c0)/2]

        self.endpoints = []

        for b in beam_points:
            beam = Beam(self.config, self.center, b)
            endpoint = beam.intersect()
            endpoint = self.bound(endpoint)

            self.endpoints.append(endpoint)

    def step(self, acceleration_value, rotation_value) -> None:
        #print(f'acceleration_value: {acceleration_value}, rotation_value: {rotation_value}')
        if self.is_dead:
            return

        rotation_value = np.clip(rotation_value, self.config.rotation_value_min, self.config.rotation_value_max)
        acceleration_value = np.clip(acceleration_value, self.config.acceleration_value_min, self.config.acceleration_value_max)

        self.steering_angle += rotation_value * self.config.rotation_ratio
        self.steering_angle = np.clip(rotation_value, self.config.min_steering_angle, self.config.max_steering_angle)

        if acceleration_value < 0:
            acceleration_value = acceleration_value * self.config.backward_acceleration_ratio
        else:
            acceleration_value = acceleration_value * self.config.forward_acceleration_ratio

        self.angle += self.steering_angle
        self.angle %= 2 * np.pi

        self.velocity += acceleration_value
        self.velocity = np.clip(self.velocity, -1, 1)

        vel_x = self.velocity * np.cos(self.angle)
        vel_y = self.velocity * np.sin(self.angle)

        step = 1

        off = Coord(vel_x*step, vel_y*step)

        self.center += off
        self.center = self.bound(self.center)

        self.update_coords()
        self.run_beams()

        #state = self.current_state()
        #norm_center = self.point_norm(self.center)
        #dist = self.dist_to_point(norm_center, self.point_norm(self.goal))
        #print(f'c: {self.center}, norm: {norm_center}, dist: {dist:.4f}, angle: {self.angle:.4f}, velocity: {self.velocity:.4f}, beam_dists: {state[4:]}')

    def point_norm(self, c):
        return Coord(c.x / float(self.config.image_width), c.y / float(self.config.image_height))

    @staticmethod
    def dist_to_point(a, p):
        return np.math.sqrt((a.x - p.x)**2 + (a.y - p.y)**2)

    def current_state(self):
        norm_center = self.point_norm(self.center)
        flat_beam_dists = []
        for ep in self.endpoints:
            ep = self.point_norm(ep)

            dist = self.dist_to_point(norm_center, ep)
            flat_beam_dists.append(dist)

        norm_goal = self.point_norm(self.goal)
        return np.array(list(norm_center.export()) + list(norm_goal.export()) + [self.angle, self.velocity] + flat_beam_dists)
        #return np.array([self.angle, self.velocity] + flat_beam_dists)
        #return np.array([self.angle, self.velocity])

    def render(self) -> None:
        colour = self.config.car_colour
        if self.is_dead:
            colour = self.config.dead_car_colour

        pygame.draw.polygon(self.window, colour, self.coords.export_int())

        if not self.is_dead:
            for ep in self.endpoints:
                pygame.draw.line(self.window, self.config.beam_colour, self.center.export_int(), ep.export_int())

class Config:
    image_path = ''
    image_map = None
    image_width = None
    image_height = None

    output_dir = '/home/zbr/awork/rl/world_model/outputs/simple_map'

    car_width = 3
    car_length = 8

    lidar_step = 2

    start_colour = (0xff, 0xee, 0x00)
    nonroad_colour = (255, 255, 255)
    car_colour = (0x21, 0x4f, 0x3b)
    beam_colour = (0x41, 0x29, 0x29)
    dead_car_colour = (0, 0, 0)
    goal_colour = (0, 0, 255)

    goal = Coord(569, 338)
    #goal = Coord(300, 50)

    obstacle_colours = [nonroad_colour, car_colour, dead_car_colour]
    road_colours = [start_colour, goal_colour]

    backward_acceleration_ratio = 0.1

    max_steering_angle = np.pi/4
    min_steering_angle = -np.pi/4

    rotation_value_abs = 10. / 180. * np.pi
    rotation_value_min = -rotation_value_abs
    rotation_value_max = rotation_value_abs

    rotation_ratio = 0.1
    backward_acceleration_ratio = 0.05
    forward_acceleration_ratio = 0.1

    acceleration_value_min = -1
    acceleration_value_max = +1

class MapGame:
    window: pygame.Surface
    bg_surf: pygame.Surface
    clock: pygame.time.Clock

    def __init__(self, config: Config):
        self.config = config
        self.window = None

        self.reset()

        self.image = PIL.Image.open(self.config.image_path)
        config.image_map = self.image

        bbox = self.image.getbbox()
        self.config.image_width = bbox[2]
        self.config.image_height = bbox[3]

        self.set_goal(config.goal)

        self.start_min, self.start_max = self.locate_start()

    def set_goal(self, goal):
        self.goal = goal

        for car in self.cars:
            car.set_goal(goal)

    def coord_is_ok(self, c: Coord) -> bool:
        return c.x < self.config.image_width and c.y < self.config.image_height and c.x >= 0 and c.y >= 0

    def set_random_goal(self):
        x = np.random.randint(self.config.image_width)
        y = np.random.randint(self.config.image_height)

        points = []

        for i in range(x, self.config.image_width, 1):
            p = self.config.image_map.getpixel((i, y))
            if not p in self.config.obstacle_colours:
                dist = i - x
                points.append((dist, Coord(i, y)))
                break

        for i in range(x, 0, -1):
            p = self.config.image_map.getpixel((i, y))
            if not p in self.config.obstacle_colours:
                dist = x - i
                points.append((dist, Coord(i, y)))
                break

        for j in range(y, self.config.image_height, 1):
            p = self.config.image_map.getpixel((x, j))
            if not p in self.config.obstacle_colours:
                dist = j - y
                points.append((dist, Coord(x, j)))
                break

        for j in range(y, 0, -1):
            p = self.config.image_map.getpixel((x, j))
            if not p in self.config.obstacle_colours:
                dist = y - j
                points.append((dist, Coord(x, j)))
                break

        for i in range(min(self.config.image_width, self.config.image_height)):
            c = Coord(x+i, y+i)
            if not self.coord_is_ok(c):
                break

            p = self.config.image_map.getpixel((c.x, c.y))
            if not p in self.config.obstacle_colours:
                dist = i
                points.append((dist, c))
                break

        for i in range(min(self.config.image_width, self.config.image_height)):
            c = Coord(x-i, y-i)
            if not self.coord_is_ok(c):
                break

            p = self.config.image_map.getpixel((c.x, c.y))
            if not p in self.config.obstacle_colours:
                dist = i
                points.append((dist, c))
                break

        for i in range(min(self.config.image_width, self.config.image_height)):
            c = Coord(x+i, y-i)
            if not self.coord_is_ok(c):
                break

            p = self.config.image_map.getpixel((c.x, c.y))
            if not p in self.config.obstacle_colours:
                dist = i
                points.append((dist, c))
                break

        for i in range(min(self.config.image_width, self.config.image_height)):
            c = Coord(x-i, y+i)
            if not self.coord_is_ok(c):
                break

            p = self.config.image_map.getpixel((c.x, c.y))
            if not p in self.config.obstacle_colours:
                dist = i
                points.append((dist, c))
                break

        min_dist = points[0][0]
        goal = points[0][1]
        for p in points[1:]:
            dist = p[0]
            g = p[1]
            if dist < min_dist:
                goal = g

        self.set_goal(goal)

    def locate_start(self) -> Tuple[Coord, Coord]:
        a = np.asarray(self.image)
        start = np.nonzero((a[:, :, 0] == self.config.start_colour[0]) & (a[:, :, 1] == self.config.start_colour[1]) & (a[:, :, 2] == self.config.start_colour[2]))
        if len(start[0]) == 0:
            raise ValueError(f'could not locate start position, start_colour: {self.config.start_colour}')

        min_x = start[1].min()
        min_y = start[0].min()
        max_x = start[1].max()
        max_y = start[0].max()

        min_c = Coord(min_x, min_y)
        max_c = Coord(max_x, max_y)
        return (min_c, max_c)

    def add_cars(self, num_cars):
        diff_x = self.start_max.x - self.start_min.x
        diff_y = self.start_max.y - self.start_min.y

        step_x = diff_x / num_cars
        step_y = diff_y / num_cars
        for i in range(num_cars):
            c = self.start_min + Coord((step_x)*i, (step_y + self.config.car_width)*i)
            self.add_car(c)

    def add_car(self, coord: Coord) -> None:
        #angle = -90
        angle = 0
        angle = angle * np.pi / 180.
        c = Car(self.config, coord, angle, self.window)
        c.set_goal(self.goal)
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

        pygame.draw.circle(self.window, self.config.goal_colour, self.goal.export_int(), 5)
        pygame.display.update()

    def reset(self) -> None:
        self.cars = []

    def current_state(self, car_number=0):
        states = []
        states.append(self.cars[car_number].current_state())

        return np.vstack(states)

    def step(self, actions) -> None:
        for car, action in zip(self.cars, actions):
            car.step(acceleration_value=action[0], rotation_value=action[1])

        for car in self.cars:
            coords = car.get_polygon()
            for c in coords:
                if c.x >= self.config.image_width or c.y >= self.config.image_height:
                    car.crash()
                    break

                try:
                    p = self.config.image_map.getpixel((c.x, c.y))
                except:
                    print(f'exception: c: {c}, width: {self.config.image_width}, height: {self.config.image_height}')
                    raise

                if p in self.config.obstacle_colours:
                    car.crash()
                    break

def main() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_image', type=str, required=True, help='Map image')
    parser.add_argument('--num_cars', type=int, default=1, help='Number of cars on the map')
    parser.add_argument('--output_dir', type=str, help='When set, save rendered frames there')
    FLAGS = parser.parse_args()

    config = Config()
    config.image_path = FLAGS.map_image

    map_game = MapGame(config)
    map_game.init_render()
    map_game.add_cars(FLAGS.num_cars)

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

        actions = [[0.01, 0.0]]*FLAGS.num_cars
        map_game.step(actions)
        map_game.render()

        if FLAGS.output_dir:
            pygame.image.save(map_game.window, os.path.join(FLAGS.output_dir, f'/stream{step:04d}.png'))
        step += 1
    pygame.quit()

if __name__ == '__main__':
    main()
