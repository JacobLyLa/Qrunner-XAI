import os
import pickle

import matplotlib.pyplot as plt
import torch

from q_network import QNetwork


def prepare_folders(folders_path):
    os.makedirs(folders_path, exist_ok=True)
    for file in os.listdir(folders_path):
        os.remove(f"{folders_path}/{file}")

def load_data():
    with open('../data/game_steps.pickle', 'rb') as f:
        game_steps = pickle.load(f)
    return game_steps

def load_q_network_device(model_path="../runs/20230927-233906/models/model_9999999.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_network = QNetwork(model_path).to(device).eval()
    return q_network, device
