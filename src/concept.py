import random

import numpy as np
from sklearn.model_selection import train_test_split

from utils import prepare_folders


class Concept:
    def __init__(self, name, binary, value_function):
        self.name = name
        self.binary = binary
        self.value_function = value_function 
        self.path = f"../concepts/{name}"

    def _prepare_binary_data(self, game_steps, train_size, test_size):
        presence = []
        absence = []
        sufficent = False
        for game_step in game_steps:
            if self.value_function(game_step.state_variables):
                presence.append(game_step)
            else:
                absence.append(game_step)
            min_length = min(len(presence), len(absence))
            # enough samples for both classes
            if min_length * 2 >= train_size + test_size:
                sufficent = True
                break

        if not sufficent:
            print(f"Concept: {self.name}, dataset size: {2 * min_length}")
            # update train_size and test_size to maximum and same ratio
            ratio = train_size / (train_size + test_size)
            total_data = 2*min_length
            train_size = int(total_data*ratio)
            test_size = total_data - train_size

        # cut off the longer list to make them equal length
        if min_length == 0:
            assert False, "No data for concept"
        presence = random.sample(presence, min_length)
        absence = random.sample(absence, min_length)

        # split the data, keeping the classes balanced
        class_train_size = train_size//2
        class_test_size = test_size//2

        presence_train = presence[:class_train_size]
        absence_train = absence[:class_train_size]
        presence_test = presence[class_train_size:class_train_size + class_test_size]
        absence_test = absence[class_train_size:class_train_size + class_test_size]

        data_train = presence_train + absence_train
        data_test = presence_test + absence_test
        y_train = [1]*len(presence_train) + [0]*len(absence_train)
        y_test = [1]*len(presence_test) + [0]*len(absence_test)

        return data_train, data_test, y_train, y_test

    def _prepare_non_binary_data(self, game_steps, train_size, test_size):        
        values = []
        for i in range(train_size + test_size):
            values.append(self.value_function(game_steps[i].state_variables))

        # scale by standard deviation to use similar regularization for all concepts
        values = np.array(values)
        values = values / np.std(values)

        # split the data
        game_steps = game_steps[:train_size + test_size]
        data_train = game_steps[:train_size]
        data_test = game_steps[train_size:]
        y_train = values[:train_size]
        y_test = values[train_size:]

        return data_train, data_test, y_train, y_test

    def prepare_data(self, game_steps, test_ratio=0.2, max_size=None):
        if not max_size:
            max_size = len(game_steps)

        train_size = int(max_size*(1-test_ratio))
        test_size = int(max_size*test_ratio)

        np.random.shuffle(game_steps)
        if self.binary:
            data_train, data_test, y_train, y_test = self._prepare_binary_data(game_steps, train_size, test_size)
        else:
            data_train, data_test, y_train, y_test = self._prepare_non_binary_data(game_steps, train_size, test_size)

        self.data_train = np.array(data_train) # can be used for forcing autoencoder to learn ball position
        self.data_test = np.array(data_test) # can be used for sorting
        self.obs_train = np.array([game_step.observation for game_step in data_train])
        self.obs_test = np.array([game_step.observation for game_step in data_test])
        self.images_train = np.array([game_step.image for game_step in data_train])
        self.images_test = np.array([game_step.image for game_step in data_test])
        self.values_train = np.array(y_train)
        self.values_test = np.array(y_test)

if __name__ == "__main__":
    from concepts import concept_instances
    from utils import load_data

    concept = concept_instances['ball distance paddle']
    data = load_data()
    print(f"Data size: {len(data)}")
    concept.prepare_data(data, max_size=500)

    # check if data is balanced
    print(f"Train Mean: {np.mean(concept.values_train)}")
    print(f"Test Mean: {np.mean(concept.values_test)}")

    # check shapes of all data
    print(concept.data_train.shape)
    print(concept.data_test.shape)
    print(concept.obs_train.shape)
    print(concept.obs_test.shape)
    print(concept.images_train.shape)
    print(concept.images_test.shape)
    print(concept.values_train.shape)
    print(concept.values_test.shape)