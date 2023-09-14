import os
import matplotlib.pyplot as plt

def prepare_folders(folders_path):
    os.makedirs(folders_path, exist_ok=True)
    for file in os.listdir(folders_path):
        os.remove(f"{folders_path}/{file}")

def plot_obs(obs):
    '''
    plot greyscale observation with size 1x4x84x84
    convert to numpy and flatten image stacks
    '''
    np_obs = obs.detach().cpu().numpy().reshape(-1, 84)
    plt.imshow(np_obs, cmap='gray')
    plt.show()