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
from agents.navigation.basic_agent import BasicAgent
from agents.tools.misc import *
from siameseCVAE import *
from velocityNN import *

from latent_agent import *
from environment import *

import random
import time
import threading
import weakref
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils # We should use this eventually.
from torch import nn, optim
from torch.nn import functional as F

TRAJECTORY_NUM = 1
SYNC = True
DEBUG = False
SPAWN_POINT_INDICES = [116,198]
AGENT = 'latent'
NUM_CAM = 1
GOAL_IMAGE = "./goal_image.png"

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


        destination_transform = spawn_points[options_dict['spawn_point_indices'][1]]

        if options_dict['agent'] == 'latent':
            transform = transforms.Compose([
                transforms.Resize((150,200)),
                transforms.ToTensor()])
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
            modelVAE = siameseCVAE(batch=1)
            checkpointVAE = torch.load('./siamese_chpt.pt')
            modelVAE.load_state_dict(checkpointVAE) 
            modelVAE.to(device)
            modelVAE.eval()

            modelVel = velocityNN()
            checkpointVel = torch.load('./velocity_single_test.pt')
            modelVel.load_state_dict(checkpointVel) 
            modelVel.to(device)
            modelVel.eval()

            models = {'VAE':modelVAE,'Vel':modelVel}

            agent = LatentAgent(vehicle, models=models, device=device) # , transform=transform
            agent.set_destination(options_dict['goal_image'],transform=transform)
        else:
            print('Going to ', destination_transform)
            agent = BasicAgent(vehicle.vehicle)
            agent.set_destination((destination_transform.location.x, destination_transform.location.y, destination_transform.location.z))

        camera_bp = ['sensor.camera.rgb', 'sensor.camera.depth', 'sensor.lidar.ray_cast']

        if options_dict['num_cam'] == 1:
            camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15, yaw=0))
            cam1 = Camera(camera_bp[0], camera_transform, vehicle, options_dict['trajector_num'], save_data=False)
        elif options_dict['num_cam'] == 2:
            camera_transform = [carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15, yaw=40)), carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15, yaw=-40)), carla.Transform(carla.Location(x=1.5, z=2.4))]
            cam1 = Camera(camera_bp[0], camera_transform[0], vehicle, options_dict['trajector_num'], save_data=False)
            cam2 = Camera(camera_bp[0], camera_transform[1], vehicle, options_dict['trajector_num'], save_data=False)

        prev_location = vehicle.vehicle.get_location()

        sp = 2

        print('Starting simulation.')
        while True:
            world.world.tick()
            world_snapshot = world.world.wait_for_tick(60.0)

            if not world_snapshot:
                continue

            # wait for sensors to sync
            # while world_snapshot.frame_count!=depth.frame_n or world_snapshot.frame_count!=segment.frame_n:
            #     time.sleep(0.05)

            control = agent.run_step()            
            vehicle.vehicle.apply_control(control)

            # check if destination reached
            current_location = vehicle.vehicle.get_location()
            # kind of hacky way to test destination reached and doesn't always work - may have to manually stop with ctrl c
            if current_location.distance(prev_location) <= 0.0 and current_location.distance(destination_transform.location) <= 10: 
                print('distance from destination: ', current_location.distance(destination_transform.location))
                # if out of destinations break else go to next destination
                if len(options_dict['spawn_point_indices']) <= sp:
                    break
                else:
                    destination_transform.location = spawn_points[options_dict['spawn_point_indices'][sp]].location
                    print('Going to ', destination_transform.location)
                    # agent.set_destination((destination_transform.location.x, destination_transform.location.y, destination_transform.location.z))
                    agent.set_destination(vehicle_transform, destination_transform)
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
        'sync': SYNC,
        'num_cam':NUM_CAM,
        'trajector_num':TRAJECTORY_NUM,
        'goal_image':GOAL_IMAGE
    }
    game_loop(options_dict)
