import os
import random
import re
import time
import argparse

import numpy as np
import torch

from src.DRL.DQN import DQN
from src.DRL.wrapped_qrunner import QrunnerWrapper, HumanRenderWrapper
from src.Qrunner.qrunner import QrunnerEnv

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
            total_reward += info['episode']['r']
            total_episodes += 1
    print(f"Average reward: {total_reward / total_episodes}")
    print(f"Max reward: {max_reward}")

def dqn_policy(model, env):
    obs, info = env.reset()
    for _ in range(10000):
        if random.random() < 0.005:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state = torch.Tensor(obs).unsqueeze(0)
                q_values = model(state)
                action = q_values.argmax(dim=1).item()
        next_obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(info)
            obs, info = env.reset()
        else:
            obs = next_obs

def main():
    model_path = "models/qrunner_dqn_20240925_094136.pth"
    model = DQN(use_dueling=True)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Loaded model from {model_path}")

    # Initialize environment with grayscale option
    env = QrunnerWrapper(
        QrunnerEnv(),
        max_steps=1819,
        max_steps_reward=156,
        blending_alpha=0.6323532669728039,
        frame_skip=4,
        use_grayscale=False
    )

    env = HumanRenderWrapper(env, scale=6, fps=30)
    dqn_policy(model, env)
    env.close()

if __name__ == "__main__":
    main()
