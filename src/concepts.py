import os
import random
from PIL import Image
import numpy as np

from utils import prepare_folders

class Concept:
    def __init__(self, name, presence_predicate, max_size=200):
        self.name = name
        self.presence_predicate = presence_predicate
        self.max_size = max_size
        self.presence_observations = []
        self.presence_images = []
        self.absence_observations = []
        self.absence_images = []

        self.path = f"../concepts/{name}"

    def check_observation(self, observation, state_variables, image):
        # check if concept is present in observation
        if self.presence_predicate(state_variables):
            self.presence_observations.append(observation)
            self.presence_images.append(image)
        else:
            self.absence_observations.append(observation)
            self.absence_images.append(image)

    def save_observations(self):
        # save presence + absence observations and images (called by state_observer)
        print(f"Saving concept {self.name}...")
        prepare_folders(self.path + "/presence_images")
        prepare_folders(self.path + "/absence_images")

        # shuffle both lists with the same order
        samples = min(self.max_size, len(self.presence_images))
        print(f"presence images: {samples}")
        self.presence_images, self.presence_observations = zip(
            *random.sample(list(zip(self.presence_images, self.presence_observations)), samples)
        )

        # same with absence lists
        samples = min(self.max_size, len(self.absence_images))
        print(f"absence images: {samples}")
        self.absence_images, self.absence_observations = zip(
            *random.sample(list(zip(self.absence_images, self.absence_observations)), samples)
        )

        # save concept images: np array (210, 160, 3)
        for i, image in enumerate(self.presence_images):
            image = Image.fromarray(image)
            image.save(f"{self.path}/presence_images/image_{i}.png")

        for i, image in enumerate(self.absence_images):
            image = Image.fromarray(image)
            image.save(f"{self.path}/absence_images/image_{i}.png")

        # save concept observations as a single numpy array
        print("saving observations...")
        np.save(f"{self.path}/presence_observations.npy", np.array(self.presence_observations))
        np.save(f"{self.path}/absence_observations.npy", np.array(self.absence_observations))

    def save_cavs(self, cavs, layers):
        prepare_folders(f"{self.path}/cavs")
        for layer, cav in zip(layers, cavs):
            np.save(f"{self.path}/cavs/cav_{layer}.npy", cav)

    def load_cav(self, layer):
        return np.load(f"{self.path}/cavs/cav_{layer}.npy")

    def get_presence_observations(self):
        return np.load(f"{self.path}/presence_observations.npy")
        
    def get_absence_observations(self):
        return np.load(f"{self.path}/absence_observations.npy")

    def get_presence_images(self):
        if len(self.presence_images) > 0:
            return np.array(self.presence_images)
        num_images = len(os.listdir(f"{self.path}/presence_images"))
        for i in range(num_images):
            image = Image.open(f"{self.path}/presence_images/image_{i}.png")
            self.presence_images.append(np.array(image))
        return np.array(self.presence_images)

    def get_absence_images(self):
        if len(self.absence_images) > 0:
            return np.array(self.absence_images)
        num_images = len(os.listdir(f"{self.path}/absence_images"))
        for i in range(num_images):
            image = Image.open(f"{self.path}/absence_images/image_{i}.png")
            self.absence_images.append(np.array(image))
        return np.array(self.absence_images)

concept_instances = [
    # TODO: more samples, and sample randomly
    Concept(name="random",
    presence_predicate=lambda state_variables: random.random() < 0.5
    ),

    Concept(name="ball above paddle",
    presence_predicate=lambda state_variables:
        abs(state_variables['player_x'] - state_variables['ball_x']) <= 8
    ),

    Concept(name="last life",
    presence_predicate=lambda state_variables: 
        state_variables['lives'] == 1
    ),

    Concept(name="ball low",
    presence_predicate=lambda state_variables: 
        state_variables['ball_y'] > 150,
    ),

    Concept(name="ball close paddle",
    presence_predicate=lambda state_variables: 
        abs(state_variables['player_x'] - state_variables['ball_x']) <= 30 and
        state_variables['ball_y'] > 160,
    ),

    Concept(name="ball left paddle",
    presence_predicate=lambda state_variables: 
        state_variables['ball_x'] < state_variables['player_x'],
    ),

    Concept(name="ball right paddle",
    presence_predicate=lambda state_variables: 
        state_variables['ball_x'] > state_variables['player_x'],
    ),

    Concept(name="paddle left corner",
    presence_predicate=lambda state_variables: 
        state_variables['player_x'] < 90,
    ),

    Concept(name="ball collision",
    presence_predicate=lambda state_variables: 
        state_variables['collision'] == True,
    ),

    Concept(name="ball fast horizontal",
    presence_predicate=lambda state_variables: 
        abs(state_variables['ball_vx']) > 10,
    ),

    Concept(name="ball fast vertical",
    presence_predicate=lambda state_variables:
        abs(state_variables['ball_vy']) > 10,
    ),

    Concept(name="lost life",
    presence_predicate=lambda state_variables:
        state_variables['lost_life'] == True,
    ),
]