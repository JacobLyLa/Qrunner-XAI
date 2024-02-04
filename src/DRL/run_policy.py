import os
import random
import re
import time

import numpy as np
import torch

from src.DRL.qnetwork import QNetwork
from src.DRL.wrapped_qrunner import wrapped_qrunner_env


def dqn_policy(model, env, compute_saliency=False):
    obs, info = env.reset()
    for _ in range(10000):
        np_obs = np.array(obs)
        batch_obs = torch.Tensor(np_obs)
        single_obs = torch.Tensor(np_obs).unsqueeze(0)

        if compute_saliency:
            single_obs.requires_grad = True

        q_values = model(single_obs)
        env.unwrapped.push_q_values(q_values.squeeze(0).detach().numpy())
        action = q_values.argmax(dim=1).item()

        # Compute saliency map if required
        if compute_saliency:
            saliency_map = compute_saliency_map(model, single_obs, None, q_values)
            #saliency_map = compute_smoothgrad_saliency_map(model, batch_obs, None)
            env.unwrapped.set_salience(saliency_map)

        next_obs, reward, terminated, truncated, info = env.step(action)
        obs = next_obs
        if terminated or truncated:
            print(info['episode'])

def compute_saliency_map(model, single_obs, action, q_values):
    if action is None:
        chosen_q_value = q_values.mean()  # Average over all actions
    else:
        chosen_q_value = q_values[0, action]  # Choose the Q-value for the chosen action

    # Backward pass to compute gradient
    model.zero_grad()
    chosen_q_value.backward()

    # First batch and only batch
    gradients = single_obs.grad[0]
    # Sum over channels (RGB and frame stacks) 
    saliency_map = gradients.abs().sum(dim=2)

    return saliency_map.cpu().numpy()

def compute_smoothgrad_saliency_map(model, batch_obs, action, num_samples=32, noise_factor=0.01):
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
    record_video = False
    human_render = True
    frame_skip = 4
    frame_stack = 4
    newest = True
    standard_path = "runs/20240125-235727/model_8000000.pt"
    
    model_path = QNetwork.find_newest_model() if newest else standard_path
    print(f"Using model: {model_path}")
    model = QNetwork(frame_stacks=frame_stack, model_path=model_path)
    
    env = wrapped_qrunner_env(frame_skip=frame_skip, frame_stack=frame_stack, record_video=record_video, human_render=human_render, scale=6)
    dqn_policy(model, env, compute_saliency=True)
    #random_policy(env)
    env.close()

if __name__ == "__main__":
    main()
