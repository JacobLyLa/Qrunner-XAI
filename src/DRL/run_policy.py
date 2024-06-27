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
    num_episodes = 1000
    
    coins_picked = []
    progression_missed1_blue = []
    progression_missed1_gold = []
    progression_picked1_red = []
    
    blue_coins = []
    red_coins = []
    gold_coins = []
    level_progression = []
    bullet_cause = 0
    lava_cause = 0
    
    player_x = 0
    cause = None
    print(f"Episode {episode}/{num_episodes}")
    while episode < num_episodes:
        single_obs = torch.Tensor(obs).unsqueeze(0)
        single_obs.requires_grad = True

        if 0.01 < random.random():
            q_values = model(single_obs)
            action = q_values.argmax(dim=1).item()
        else:
            action = random.randint(0, 4)

        # Send gradient
        if send_gradients:
            gradient = get_gradient(model, single_obs, None, q_values)
            #env.unwrapped.push_q_values(q_values.squeeze(0).detach().numpy())
            #gradient = get_smooth_gradient(model, torch.Tensor(obs), None)
            env.unwrapped.set_gradient(gradient)
        else:
            model.eval()

        next_obs, reward, terminated, truncated, info = env.step(action)
        if env.cause != cause:
            cause = env.cause
            if env.cause == "bullet":
                bullet_cause += 1
            elif env.cause == "lava":
                lava_cause += 1
        if terminated or truncated:
            # Add metric if missing
            if episode == len(progression_missed1_blue):
                progression_missed1_blue.append(player_x)
            if episode == len(progression_missed1_gold):
                progression_missed1_gold.append(player_x)
            if episode == len(progression_picked1_red):
                progression_picked1_red.append(player_x)
            
            episode += 1
            if episode % 10 == 0 or episode == num_episodes:
                print(f"Episode {episode}/{num_episodes}")
            blue = coins_picked.count("blue")
            red = coins_picked.count("red")
            gold = coins_picked.count("gold")
            level_progression.append(player_x)
            blue_coins.append(blue)
            red_coins.append(red)
            gold_coins.append(gold)
        else:
            coins_picked = env.player.coins_picked.copy()
            player_x = env.player.x
            
            blue_coins_missed = env.player.coins_missed.count("blue")
            if episode == len(progression_missed1_blue):
                if blue_coins_missed == 1:
                    progression_missed1_blue.append(player_x)
                    
            gold_coins_missed = env.player.coins_missed.count("gold")
            if episode == len(progression_missed1_gold):
                if gold_coins_missed == 1:
                    progression_missed1_gold.append(player_x)
                    
            red_coins_picked = env.player.coins_picked.count("red")
            if episode == len(progression_picked1_red):
                if red_coins_picked == 1:
                    progression_picked1_red.append(player_x)
            
        obs = next_obs
    
    print(f"bullet cause: {bullet_cause}/{num_episodes}")
    print(f"lava cause: {lava_cause}/{num_episodes}")
    print(f"timeout: {num_episodes - bullet_cause - lava_cause}/{num_episodes}")
    
    print(f"blue coins mean and std: {np.mean(blue_coins), np.std(blue_coins)}")
    print(f"red coins mean and std: {np.mean(red_coins), np.std(red_coins)}")
    print(f"gold coins mean and std: {np.mean(gold_coins), np.std(gold_coins)}")
    print(f"level progression mean and std: {np.mean(level_progression), np.std(level_progression)}")
    
    print(f"progression after missed blue coin mean and std: {np.mean(progression_missed1_blue), np.std(progression_missed1_blue)}")
    print(f"progression after missed gold coin mean and std: {np.mean(progression_missed1_gold), np.std(progression_missed1_gold)}")
    print(f"progression after picked red coin mean and std: {np.mean(progression_picked1_red), np.std(progression_picked1_red)}")

def get_gradient(model, single_obs, action, q_values):
    if action is None:
        chosen_q_value = q_values.mean() # Average over all actions
    else:
        chosen_q_value = q_values[0, action] # Choose the Q-value for the chosen action

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
        chosen_q_values = q_values.mean(dim=0) # Average over all actions
    else:
        chosen_q_values = q_values[:, action] # Select the Q-values for the chosen action

    model.zero_grad()
    chosen_q_values.sum().backward()

    # Extract and average gradients
    gradients = noisy_samples.grad # (num_samples, C, H, W)
    saliency_map = gradients.abs().mean(dim=3).sum(dim=0) # Average over samples and sum over channels

    return saliency_map.cpu().numpy()
    
def random_policy(env):
    obs, info = env.reset()
    total_reward = 0
    total_episodes = 0
    max_reward = 0
    for _ in range(10000):
        action = 2 if random.random() < 0.8 else 3
        obs, rewards, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            if info['episode']['r'] > max_reward:
                max_reward = info['episode']['r']
            # print(info['episode'])
            total_reward += info['episode']['r']
            total_episodes += 1
    print(f"Average reward: {total_reward / total_episodes}")
    print(f"Max reward: {max_reward}")

def main():
    original_env = True
    render_human = False
    render_salient = False
    record_video = False
    plot_q = False
    newest = False
    frame_skip = 4
    #standard_path = "runs/20240317-112025/model_10000000.pt"
    standard_path = "runs/20240322-100656/model_5000000.pt"
    
    # 2025 E
    # 3500 E->E'
    # 0656 E->E'->E
    
    model_path = QNetwork.find_newest_model() if newest else standard_path
    model = QNetwork(model_path=model_path)
    
    env = wrapped_qrunner_env(frame_skip=frame_skip, human_render=render_human, render_salient=render_salient, plot_q=plot_q, record_video=record_video, scale=6, original=original_env)
    dqn_policy(model, env, send_gradients=render_salient)
    #random_policy(env)
    env.close()

if __name__ == "__main__":
    main()
