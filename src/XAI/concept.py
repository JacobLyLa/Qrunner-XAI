import os
import pickle
import random

import numpy as np
import torch


class Concept:
    def __init__(self, name, binary, concept_function):
        self.name = name
        self.binary = binary
        self.concept_function = concept_function 
        self.folder_path = f"concepts/{name}/"
        
        # If folder does not exist, create it
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
            
    def summary(self):
        print('Concept:', self.name)
        print('binary:', self.binary)
        print('train:', self.train_obs.shape, self.train_values.shape)
        print('test:', self.test_obs.shape, self.test_values.shape)
        
    def save_torch_probe(self, probe, model_name, layer, score):
        # Remove .pt files in folder
        files = os.listdir(self.folder_path)
        for file in files:
            if file.endswith(".pt"):
                os.remove(os.path.join(self.folder_path, file))
        torch.save(probe, f"{self.folder_path}{model_name}-{layer}-{round(score,3)}.pt")
        
    def load_torch_probe(self):
        # Find only .pt file in folder
        files = os.listdir(self.folder_path)
        for file in files:
            if file.endswith(".pt"):
                score = float(file.split("-")[-1].split(".pt")[0])
                return torch.load(os.path.join(self.folder_path, file)), score
        return None, 0
        
    def _prepare_binary_data(self, env_steps, train_size, test_size):
        presence = []
        absence = []
        sufficent_samples = False
        # Append game steps to presence or absence until sufficent samples
        for game_step in env_steps:
            if self.concept_function(game_step.state_variables):
                presence.append(game_step)
            else:
                absence.append(game_step)
            min_length = min(len(presence), len(absence))
            # Enough samples for both classes
            if min_length * 2 >= train_size + test_size:
                sufficent_samples = True
                break

        if not sufficent_samples:
            # Update train_size and test_size to maximum and same ratio
            ratio = train_size / (train_size + test_size)
            total_data = 2 * min_length
            train_size = int(total_data*ratio)
            test_size = total_data - train_size

        # Slice the longer list to make them equal length
        assert min_length != 0, f"[{self.name}] No samples for one of the classes"
        presence = random.sample(presence, min_length)
        absence = random.sample(absence, min_length)

        # Split the data, keeping the classes balanced
        class_train_size = train_size//2
        class_test_size = test_size//2

        presence_train = presence[:class_train_size]
        absence_train = absence[:class_train_size]
        presence_test = presence[class_train_size:class_train_size + class_test_size]
        absence_test = absence[class_train_size:class_train_size + class_test_size]

        train_data = presence_train + absence_train
        test_data = presence_test + absence_test
        y_train = [1]*len(presence_train) + [0]*len(absence_train)
        y_test = [1]*len(presence_test) + [0]*len(absence_test)

        return train_data, test_data, y_train, y_test

    def _prepare_non_binary_data(self, env_steps, train_size, test_size):        
        values = []
        for i in range(train_size + test_size):
            values.append(self.concept_function(env_steps[i].state_variables))
        values = np.array(values)

        # Train test split
        env_steps = env_steps[:train_size + test_size]
        train_data = env_steps[:train_size]
        test_data = env_steps[train_size:]
        y_train = values[:train_size]
        y_test = values[train_size:]

        return train_data, test_data, y_train, y_test

    def prepare_data(self, env_steps, test_ratio=0.2, max_size=None):
        '''
        Prepare data for the concept
        Access the data through train_/test_ + obs/images/values
        Obs: network input
        Images: network input in human readable image format
        Values: concept values
        '''
        if not max_size or max_size > len(env_steps):
            max_size = len(env_steps)

        train_size = int(max_size*(1-test_ratio))
        test_size = int(max_size*test_ratio)

        np.random.shuffle(env_steps)
        if self.binary:
            train_data, test_data, y_train, y_test = self._prepare_binary_data(env_steps, train_size, test_size)
        else:
            train_data, test_data, y_train, y_test = self._prepare_non_binary_data(env_steps, train_size, test_size)

        self.train_data = np.array(train_data)
        self.test_data = np.array(test_data)
        self.train_obs = np.array([game_step.observation for game_step in train_data])
        self.test_obs = np.array([game_step.observation for game_step in test_data])
        self.train_images = np.array([game_step.image for game_step in train_data])
        self.test_images = np.array([game_step.image for game_step in test_data])
        self.train_values = np.array(y_train)
        self.test_values = np.array(y_test)
        
    @staticmethod
    def load_concept_data():
        with open('data/env_steps.pickle', 'rb') as f:
            env_steps = pickle.load(f)
        return env_steps