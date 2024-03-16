import os
import random
import re
import time

import numpy as np
import torch

from src.DRL.qnetwork import QNetwork
from src.DRL.wrapped_qrunner import wrapped_qrunner_env

def dqn_policy(model, env):
    episode = 0
    obs, info = env.reset()
    for _ in range(10000):
        single_obs = torch.Tensor(obs).unsqueeze(0)

        single_obs.requires_grad = True

        q_values = model(single_obs)
        env.unwrapped.push_q_values(q_values.squeeze(0).detach().numpy())
        action = q_values.argmax(dim=1).item()

        # Sends gradient
        gradient = get_gradient(model, single_obs, None, q_values)
        #gradient = get_smooth_gradient(model, torch.Tensor(obs), None)
        env.unwrapped.set_gradient(gradient)

        next_obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            episode += 1
            print(info['episode'])
        obs = next_obs

def get_gradient(model, single_obs, action, q_values):
    if action is None:
        chosen_q_value = q_values.mean()  # Average over all actions
    else:
        chosen_q_value = q_values[0, action]  # Choose the Q-value for the chosen action

    # Backward pass to compute gradient
    model.zero_grad()
    chosen_q_value.backward()

    return single_obs.grad[0].cpu().numpy()

def get_smooth_gradient(model, batch_obs, action, num_samples=64, noise_factor=0.01):
    # Generate noisy samples
    noise = torch.randn((num_samples,) + batch_obs.shape) * noise_factor * 255
    noisy_samples = batch_obs + noise
    noisy_samples = noisy_samples.clamp(0, 255)

    # Forward pass
    noisy_samples = noisy_samples.detach()
    noisy_samples.requires_grad = True
    q_values = model(noisy_samples)

    if action is None:
        chosen_q_values = q_values.mean(dim=0)  # Average over all actions
    else:
        chosen_q_values = q_values[:, action]  # Select the Q-values for the chosen action

    # Backward pass
    model.zero_grad()
    chosen_q_values.sum().backward()

    # Extract and average gradients
    gradients = noisy_samples.grad  # Shape: (num_samples, C, H, W)
    saliency_map = gradients.abs().mean(dim=3).sum(dim=0)  # Average over samples and sum over channels

    return saliency_map.cpu().numpy()
    
def random_policy(env):
    obs, info = env.reset()
    total_reward = 0
    total_episodes = 0
    for _ in range(5000):
        action = 2 if random.random() < 0.8 else 3
        obs, rewards, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(info['episode'])
            total_reward += info['episode']['r']
            total_episodes += 1
    print(f"Average reward: {total_reward / total_episodes}")

def main():
    render_human = True
    render_salient = False
    record_video = False
    plot_q = False
    newest = False
    frame_skip = 4
    standard_path = "runs/20240224-103820_task_0/model_10000000.pt"
    
    model_path = QNetwork.find_newest_model() if newest else standard_path
    model = QNetwork(model_path=model_path)
    
    env = wrapped_qrunner_env(frame_skip=frame_skip, human_render=render_human, render_salient=render_salient, plot_q=plot_q, record_video=record_video, scale=6)
    dqn_policy(model, env)
    #random_policy(env)
    env.close()

if __name__ == "__main__":
    main()
