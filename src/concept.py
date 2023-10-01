import math
import os
import random

import numpy as np
from PIL import Image
from sklearn import linear_model
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from utils import prepare_folders


class Concept:
    def __init__(self, name, binary, value_function):
        self.name = name
        self.binary = binary
        self.value_function = value_function 
        self.path = f"../concepts/{name}"

    def prepare_data(self, game_steps, test_ratio=0.2, max_size=None):
        if max_size is None:
            max_size = len(game_steps)
        presence = []
        absence = []
        values = []
        if self.binary:
            for game_step in game_steps:
                if self.value_function(game_step.state_variables):
                    presence.append(game_step)
                else:
                    absence.append(game_step)
            min_length = min(len(presence), len(absence))
            if min_length == 0:
                print(f"Concept {self.name} has 0 instances of one class.")
                return
            presence = random.sample(presence, min_length)
            absence = random.sample(absence, min_length)
            game_steps = presence + absence
            values = [1] * len(presence) + [0] * len(absence)

        else:
            for game_step in game_steps:
                values.append(self.value_function(game_step.state_variables))

        # split data into train and test
        data_train, data_test, y_train, y_test = train_test_split(
            game_steps, values, test_size=test_ratio, random_state=0, shuffle=True
        )

        # seperate data into observations and images
        train_size = int(max_size*(1-test_ratio))
        test_size = int(max_size*test_ratio)
        self.data_test = data_test[:test_size] # can be used for sorting
        self.obs_train = np.array([game_step.observation for game_step in data_train])[:train_size]
        self.obs_test = np.array([game_step.observation for game_step in data_test])[:test_size]
        self.images_train = np.array([game_step.image for game_step in data_train])[:train_size]
        self.images_test = np.array([game_step.image for game_step in data_test])[:test_size]
        self.values_train = np.array(y_train)[:train_size]
        self.values_test = np.array(y_test)[:test_size]

        '''
        if not self.binary:
            # divide values by train std
            std = np.std(self.values_train)
            self.values_train = self.values_train.astype(float) / std
            self.values_test = self.values_test.astype(float) / std
        '''