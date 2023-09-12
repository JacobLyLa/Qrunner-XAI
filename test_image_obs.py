import os
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from concepts import concept_instances

concept = concept_instances[3]
print(concept.name)

# get presence images and observations
presence_observations = concept.get_presence_observations()
presence_images = concept.get_presence_images()
index = random.randint(0, len(presence_observations) - 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# plot the first image on the left subplot
axes[0].imshow(presence_images[index])
axes[0].set_title("Presence Image")

# plot the last channel (newest frame) of the observation
observation = presence_observations[index][-1]
axes[1].imshow(observation, cmap="gray")
axes[1].set_title("Observation (last frame)")

plt.show()
