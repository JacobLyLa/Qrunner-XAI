import random

import numpy as np
import torch
import os

from custom_env import make_env
from utils import load_q_network_device

if __name__ == '__main__':
    play_iterations = 50_000
    num_data_points = 5_000
    epsilon = 0.02
    
    human = False
    if human:
        env = make_env(render_mode="human")
    else:
        env = make_env(save_interval=play_iterations//num_data_points)

    q_network, device = load_q_network_device()
    obs, info = env.reset()
    for i in range(play_iterations):
        if i % 1000 == 0:
            print(i)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            np_obs = np.array(obs)
            tensor_obs = torch.Tensor(np_obs).to(device)
            reshaped_obs = tensor_obs.unsqueeze(0)
            q_values = q_network(reshaped_obs)
            action = q_values.argmax(dim=1).item()

        next_obs, reward, terminated, truncated, info = env.step(action)
        obs = next_obs
    env.save_data()
    env.close()