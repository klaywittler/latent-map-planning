#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys

try:
    sys.path.append(glob.glob('carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from agents.navigation.roaming_agent import RoamingAgent
from agents.navigation.basic_agent import BasicAgent

import random
import time
import threading
# import pygame
import weakref
from environment import *


def game_loop():
    world = None

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        if client.get_world() != 1230765172743732701:
            client.load_world('Town05') 

        world = World(client.get_world())

        # settings = world.world.get_settings()
        # settings.fixed_delta_seconds = 0.05
        # settings.synchronous_mode = True
        # world.world.apply_settings(settings)


        vehicle_bp = 'model3'
        vehicle_transform = random.choice(world.map.get_spawn_points()) # carla.Transform(carla.Location(x=48.0, y=8.001, z=0.5), carla.Rotation(yaw=-177))
        print(vehicle_transform)
        vehicle = Car(vehicle_bp, vehicle_transform, world)

        agent = agent = BasicAgent(vehicle.vehicle)
        # spawn_point = world.map.get_spawn_points()[0]
        # agent.set_destination((spawn_point.location.x, spawn_point.location.y, spawn_point.location.z))
        spawn_point = carla.Transform(carla.Location(x=49.0001, y=7.998, z=0.5), carla.Rotation(yaw=90))
        print(spawn_point)
        agent.set_destination((spawn_point.location.x, spawn_point.location.y, spawn_point.location.z))
        
        camera_bp = ['sensor.camera.rgb', 'sensor.camera.rgb', 'sensor.lidar.ray_cast']
        camera_transform = [carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15, yaw=40)), carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15, yaw=-40)), carla.Transform(carla.Location(x=1.5, z=2.4))]

        cam1 = Camera(camera_bp[0], camera_transform[0], vehicle)
        cam2 = Camera(camera_bp[1], camera_transform[1], vehicle)
        lidar = Lidar(camera_bp[2], camera_transform[2], vehicle)

        file = open("control_input.txt", "a")
        while True:
            if not world.world.wait_for_tick(10.0):
                continue

            control = agent.run_step()
            w = str(0) + ',' + str(control.steer) + ',' + str(control.throttle) + ',' + str(control.brake) + '\n'
            file.write(w)
            control.manual_gear_shift = False
            vehicle.vehicle.apply_control(control)

            world.world.tick()

    finally:
        if world is not None:
            world.destroy()


if __name__ == '__main__':

    # main()
    game_loop()
