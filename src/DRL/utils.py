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

def load_q_network_device(model_path="../runs/20231107-224748/models/model_10000000.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_network = QNetwork(model_path).to(device).eval()
    return q_network, device
