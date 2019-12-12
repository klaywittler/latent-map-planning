#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles.
The agent also responds to traffic lights. """


import carla
from agents.navigation.agent import Agent, AgentState
from agents.navigation.controller import *


import math
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils # We should use this eventually.
from torch import nn, optim
from torch.nn import functional as F

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

class LatentAgent(Agent):
    """
    BasicAgent implements a basic agent that navigates scenes to reach a given
    target destination. This agent respects traffic lights and other vehicles.
    """

    def __init__(self, vehicle, models=None, device='cpu', target_speed=50):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        super(LatentAgent, self).__init__(vehicle.vehicle)

        self.modelVAE = models['VAE']
        self.modelVel = models['Vel']    

        self._proximity_threshold = 10.0  # meters
        self._state = AgentState.NAVIGATING
        self._vehicle = vehicle

        args_longitudinal = {'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0, 'dt': 1.0/5.0}

        self._lon_controller = PIDLongitudinalController(self._vehicle.vehicle, **args_longitudinal)
        self._target_speed = target_speed
        self.device = device

    def set_destination(self, target_img, transform=None, load_as_grayscale=True, show_img=False):
        """
        This method creates a list of waypoints from agent's position to destination location
        based on the route returned by the global router
        """
        pil_img = Image.open(target_img)
        if load_as_grayscale:
            pil_img = pil_img.convert('L')

        self.transform = transform
        self.end_img = self.transform(pil_img).float().to(self.device).unsqueeze(0)
        
        # if transform:
        #     self.transform = transform
        #     transform_result = self.transform(pil_img)
        #     self.end_img = np.asarray(transform_result[:3, :, :])
        # else:
        #     self.end_img = np.array(pil_img)

        if show_img:
            cv2.imshow("goal", self.end_img)
            cv2.waitKey(1)

    def _inference(self, current_img, end_img, show_img=False):
        if show_img:
            cv2.imshow("current", current_img)
            cv2.waitKey(1)


        current_img = self.transform(current_img).float().to(self.device).unsqueeze(0)

        # steering = np.clip(np.random.normal(0.0, 1, size=1), -1.0, 1.0).item()
        # velocity = np.random.normal(5, 5, size=1).item()
        # # steering, velocity = self.model.forward(current_img, end_img)
        with torch.no_grad():
            xhat, yhat, z, z_mean, z_logvar = self.modelVAE.forward(current_img, end_img)
            vel, steer = self.modelVel.forward(z_mean)

        speed = np.linspace(0,50,6)
        angle = np.linspace(-1,1,11)
        # _, predictedVel = torch.max(vel, 1)
        # _, predictedSteer = torch.max(steer, 1)

        vel = F.softmax(vel)
        steer = F.softmax(steer)

        print('velocity output ', torch.max(vel, 1))

        velocity = sum(vel.squeeze(0).cpu().numpy()*speed)
        steering = sum(steer.squeeze(0).cpu().numpy()*angle)

        steering = np.clip(steering, -1.0, 1.0).item()

        print(f'velocity: {velocity}, steering: {steering}')

        return steering, velocity


    def run_step(self, debug=False):
        """
        Execute one step of navigation.
        :return: carla.VehicleControl
        """
        current_img = self._vehicle.image
        if current_img is not None:
            steering, velocity = self._inference(current_img, self.end_img)

            control = carla.VehicleControl()
            control.steer = steering
            control.throttle = self._lon_controller.run_step(velocity)
            control.brake = 0.0
            control.hand_brake = False
            control.manual_gear_shift = False
        else:
            control = self.emergency_stop()

        return control

