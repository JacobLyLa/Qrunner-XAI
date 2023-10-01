import os
import pickle

import matplotlib.pyplot as plt


def prepare_folders(folders_path):
    os.makedirs(folders_path, exist_ok=True)
    for file in os.listdir(folders_path):
        os.remove(f"{folders_path}/{file}")

def load_data():
    with open('../data/game_steps.pickle', 'rb') as f:
        game_steps = pickle.load(f)
    return game_steps