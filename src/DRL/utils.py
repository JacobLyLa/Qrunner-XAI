import os
import pickle

import matplotlib.pyplot as plt
import torch

from q_network import QNetwork

# TODO: move around

def load_game_data():
    with open('../data/game_steps.pickle', 'rb') as f:
        game_steps = pickle.load(f)
    return game_steps

