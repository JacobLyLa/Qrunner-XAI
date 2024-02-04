import os
import random

import numpy as np
import torch
from tqdm import tqdm

from src.DRL.qnetwork import QNetwork
from src.DRL.wrapped_qrunner import wrapped_qrunner_env
from src.XAI.state_extractor import StateExtractorWrapper

if __name__ == '__main__':
    play_iterations = 50_000
    num_data_points = 10_000
    epsilon = 0.05 # Probability of taking a random action
    # TODO: Given a model path, let model save hyperparams like frame_stack, frame_skip, etc.
    record_video = False
    human_render = False
    frame_skip = 4
    frame_stack = 4
    newest = True
    standard_path = "runs/20240125-235727/model_8000000.pt"
    
    model_path = QNetwork.find_newest_model() if newest else standard_path
    print(f"Using model: {model_path}")
    model = QNetwork(frame_stacks=frame_stack, model_path=model_path)
    
    env = wrapped_qrunner_env(frame_skip=frame_skip, frame_stack=frame_stack, record_video=record_video, human_render=human_render, scale=6)
    env = StateExtractorWrapper(env, save_interval=play_iterations//num_data_points)
    obs, info = env.reset()
    for i in tqdm(range(play_iterations)):
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            np_obs = np.array(obs)
            tensor_obs = torch.Tensor(np_obs).unsqueeze(0)
            q_values = model(tensor_obs)
            action = q_values.argmax(dim=1).item()

        obs, reward, terminated, truncated, info = env.step(action)
    env.save_data()
    env.close()