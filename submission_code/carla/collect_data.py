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
from agents.tools.misc import get_speed

import random
import time
import threading
import weakref
import numpy as np
from environment import *

TRAJECTORY_NUM = 6

DEBUG = False
SYNC = True
SPAWN_POINT_INDICES = [116,198]
AGENT = 'basic'
NUM_CAM = 1

def game_loop(options_dict):
    world = None

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(60.0)

        print('Changing world to Town 5')
        client.load_world('Town05') 

        world = World(client.get_world(), options_dict['sync'])
        spawn_points = world.world.get_map().get_spawn_points()

        vehicle_bp = 'model3'
        vehicle_transform = spawn_points[options_dict['spawn_point_indices'][0]]
        # vehicle_transform.location.x -= 60
        vehicle_transform.location.x -= (20 + np.random.normal(0.0, 20, size=1).item()) # start 80 std 20 # start 40 std 15 # start 10 std 10
        
        vehicle = Car(vehicle_bp, vehicle_transform, world)

        ticks = 0
        while ticks < 60:
            world.world.tick()
            world_snapshot = world.world.wait_for_tick(10.0)
            if not world_snapshot:
                continue
            else:
                ticks += 1

        agent = BasicAgent(vehicle.vehicle)
        
        destination_point = spawn_points[options_dict['spawn_point_indices'][1]].location

        print('Going to ', destination_point)
        agent.set_destination((destination_point.x, destination_point.y, destination_point.z))
        
        camera_bp = ['sensor.camera.rgb', 'sensor.camera.depth', 'sensor.lidar.ray_cast']
        

        if options_dict['num_cam'] == 1:
            camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15, yaw=0))
            cam1 = Camera(camera_bp[0], camera_transform, vehicle, options_dict['trajector_num'], save_data=True)
        elif options_dict['num_cam'] == 2:
            camera_transform = [carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15, yaw=40)), carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15, yaw=-40)), carla.Transform(carla.Location(x=1.5, z=2.4))]
            cam1 = Camera(camera_bp[0], camera_transform[0], vehicle, options_dict['trajector_num'], save_data=True)
            cam2 = Camera(camera_bp[0], camera_transform[1], vehicle, options_dict['trajector_num'], save_data=True)
        # depth1 = Camera(camera_bp[1], camera_transform[0], vehicle)
        # depth2  = Camera(camera_bp[1], camera_transform[1], vehicle)
        # lidar = Lidar(camera_bp[2], camera_transform[2], vehicle)

        prev_location = vehicle.vehicle.get_location()

        sp = 2

        file = open("control_input.txt", "a")
        while True:
            world.world.tick()
            world_snapshot = world.world.wait_for_tick(60.0)

            if not world_snapshot:
                continue

            # wait for sensors to sync
            # while world_snapshot.frame_count!=depth.frame_n or world_snapshot.frame_count!=segment.frame_n:
            #     time.sleep(0.05)

            control = agent.run_step()
            control.steer = np.clip(control.steer+np.random.normal(0.0, 0.5, size=1), -1.0, 1.0).item()
            control.throttle = np.clip(control.throttle+np.random.normal(0.5, 0.25, size=1), 0.0, 1.0).item()
            
            vehicle.vehicle.apply_control(control)

            w = str(options_dict['trajector_num']) + ',' + str(world_snapshot.frame_count) + ',' + str(control.steer) + ',' + str(get_speed(vehicle.vehicle))  + '\n'
            file.write(w)

            current_location = vehicle.vehicle.get_location()

            if current_location.distance(prev_location) <= 0.0 and current_location.distance(destination_point) <= 10:
                print('distance from destination: ', current_location.distance(destination_point))
                if len(options_dict['spawn_point_indices']) <= sp:
                    break
                else:
                    destination_point = spawn_points[options_dict['spawn_point_indices'][sp]].location
                    print('Going to ', destination_point)
                    agent.set_destination((destination_point.x, destination_point.y, destination_point.z))
                    sp += 1

            prev_location = current_location

    finally:
        if world is not None:
            world.destroy()


if __name__ == '__main__':
    sensor_dict = {
        'IM_WIDTH': 400,
        'IM_HEIGHT': 300,
        'SENSOR_TICK': 0.2,
        'FOV': 120
    }
    
    sensor_attributes(sensor_dict)

    options_dict = {
        'agent': AGENT,
        'spawn_point_indices': SPAWN_POINT_INDICES,
        'debug': DEBUG,
        'num_cam':NUM_CAM,
        'sync': SYNC,
        'trajector_num':TRAJECTORY_NUM
    }
    game_loop(options_dict)
