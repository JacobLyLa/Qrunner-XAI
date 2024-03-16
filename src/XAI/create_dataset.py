import argparse
import random
import numpy as np
import torch
from tqdm import tqdm

from src.DRL.qnetwork import QNetwork
from src.DRL.wrapped_qrunner import wrapped_qrunner_env
from src.XAI.state_extractor import StateExtractorWrapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create dataset from QRunner environment.")
    parser.add_argument('--steps', type=int, help='Number of iterations to play', default=500_000)
    parser.add_argument('--samples', type=int, help='Number of data points to save', default=20_000)
    parser.add_argument('--epsilon', type=float, help='Epsilon for epsilon-greedy policy', default=0.05)
    parser.add_argument('--frame_skip', type=int, help='Number of frames to skip', default=4)
    parser.add_argument('--newest', action='store_true', help='Load the newest model by default off')
    parser.add_argument('--model_path', type=str, help='Path to the model file', default='runs/20240224-103820_task_0/model_10000000.pt')
    args = parser.parse_args()

    model_path = QNetwork.find_newest_model() if args.newest else args.model_path
    model = QNetwork(model_path=model_path)
    
    env = wrapped_qrunner_env(frame_skip=args.frame_skip)
    env = StateExtractorWrapper(env, save_interval=args.steps//args.samples)
    obs, info = env.reset()

    for i in tqdm(range(args.steps)):
        if random.random() < args.epsilon:
            action = env.action_space.sample()
        else:
            tensor_obs = torch.Tensor(obs).unsqueeze(0)
            q_values = model(tensor_obs)
            action = q_values.argmax(dim=1).item()

        obs, reward, terminated, truncated, info = env.step(action)
    env.save_data()
    env.close()
