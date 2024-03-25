import os
import random
import re
import time

import numpy as np
import torch

from src.DRL.qnetwork import QNetwork
from src.DRL.wrapped_qrunner import wrapped_qrunner_env

def dqn_policy(model, env, send_gradients=False):
    obs, info = env.reset()
    episode = 0
    num_episodes = 100
    
    blue_coins = []
    red_coins = []
    gold_coins = []
    level_progression = []
    
    coins_picked = []
    player_x = 0
    while episode < num_episodes:
        single_obs = torch.Tensor(obs).unsqueeze(0)

        single_obs.requires_grad = True

        q_values = model(single_obs)
        action = q_values.argmax(dim=1).item()

        # Sends gradient
        if send_gradients:
            gradient = get_gradient(model, single_obs, None, q_values)
            #env.unwrapped.push_q_values(q_values.squeeze(0).detach().numpy())
            #gradient = get_smooth_gradient(model, torch.Tensor(obs), None)
            env.unwrapped.set_gradient(gradient)

        next_obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            episode += 1
            #print(info['episode'])
            blue = coins_picked.count("blue")
            red = coins_picked.count("red")
            gold = coins_picked.count("gold")
            print(f"Blue: {blue}, Red: {red}, Gold: {gold}, Level progression: {player_x}")
            level_progression.append(player_x)
            blue_coins.append(blue)
            red_coins.append(red)
            gold_coins.append(gold)
        else:
            coins_picked = env.player.coins_picked.copy()
            player_x = env.player.x
        obs = next_obs
    print(f"Blue: {blue_coins}")
    print(f"Red: {red_coins}")
    print(f"Gold: {gold_coins}")
    print(f"Level progression: {level_progression}")
    
    print(f"blue coins mean and std: {np.mean(blue_coins), np.std(blue_coins)}")
    print(f"red coins mean and std: {np.mean(red_coins), np.std(red_coins)}")
    print(f"gold coins mean and std: {np.mean(gold_coins), np.std(gold_coins)}")
    print(f"level progression mean and std: {np.mean(level_progression), np.std(level_progression)}")

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
    original_env = True
    render_human = True
    render_salient = True
    record_video = False
    plot_q = False
    newest = False
    frame_skip = 4
    standard_path = "runs/20240320-134334_task_0/model_1400000.pt"
    
    model_path = QNetwork.find_newest_model() if newest else standard_path
    model = QNetwork(model_path=model_path)
    
    env = wrapped_qrunner_env(frame_skip=frame_skip, human_render=render_human, render_salient=render_salient, plot_q=plot_q, record_video=record_video, scale=6, original=original_env)
    dqn_policy(model, env, send_gradients=render_salient)
    #random_policy(env)
    env.close()

if __name__ == "__main__":
    main()
