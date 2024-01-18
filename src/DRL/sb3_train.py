import time

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium.envs.registration import register
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env

from src.DRL.wrapped_qrunner import wrapped_qrunner_env


def get_q_values(model, obs):
    obs_tensor = th.tensor(obs, dtype=th.float32)
    # Reshape/permute the tensor to [batch_size, channels, height, width]
    obs_tensor = obs_tensor.unsqueeze(0)
    device = next(model.policy.parameters()).device
    obs_tensor = obs_tensor.to(device)
    with th.no_grad():
        q_values_tensor = model.q_net(obs_tensor)
    q_values = q_values_tensor.cpu().numpy().squeeze()
    return q_values

def create_or_load_model(algorithm, env, restart, model_name, tensorboard_log):
    if restart:
        if algorithm is DQN:
            print("Creating new DQN model")
            model = algorithm("CnnPolicy", env, verbose=0, tensorboard_log=tensorboard_log, learning_starts=30000, buffer_size=100000, gamma=0.99)
        else:
            print("Creating new PPO model")
            model = algorithm("CnnPolicy", env, verbose=0, tensorboard_log=tensorboard_log)
    else:
        model = algorithm.load(model_name)
        model.set_env(env)
    return model

def train_model(use_dqn, env, its, restart=True):
    algorithm = DQN if use_dqn else PPO
    tensorboard_log = "./tensorboard/CnnPolicy_qrunner/"
    model_name = "dqn_qrunner" if use_dqn else "ppo_qrunner"

    model = create_or_load_model(algorithm, env, restart, model_name, tensorboard_log)

    for i in range(10):
        reset_timesteps = True if i == 0 and restart else False
        model.learn(total_timesteps=its//10, reset_num_timesteps=reset_timesteps, progress_bar=True)
        model.save(model_name)
        print(f"[{i}] Model saved as {model_name}")

    return model


def main():
    frame_skip = 4
    frame_stack = 4
    env = wrapped_qrunner_env(size=84, frame_skip=frame_skip, frame_stack=frame_stack, render_mode='rgb_array', record_video=False) # for some reason record video only works for PPO and not DQN
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    check_env(env, warn=True)
    
    #model = train_model(True, env, 4000000, restart=True)
    model = DQN.load("dqn_qrunner")
    #model = train_model(False, env, 500000, restart=True)
    #model = PPO.load("ppo_qrunner")

    # Test the trained agent
    # With human render the original env and size is used.
    # Can decrease frame_skip so the agent makes decision every frame in contrast to every 4th frame
    env = wrapped_qrunner_env(size=84*10, frame_skip=1, frame_stack=frame_stack, render_mode='human', record_video=False)
    model.set_env(env)
    obs, info = env.reset()
    last_time = time.time()
    target_fps = 50
    for _ in range(10000):
        current_time = time.time()
        time.sleep(max(0, 1/target_fps - (current_time - last_time)))
        last_time = current_time
        
        action, _states = model.predict(obs, deterministic=False)
        # Print Q-values if DQN
        if hasattr(model, 'q_net'):
            q_values = get_q_values(model, obs)
            print(f"Action: {action}, Q-values: {[f'{q:.2f}' for q in q_values]}, Max Q-value: {np.max(q_values):.2f}, Mean Q-value: {np.mean(q_values):.2f}")
        
        obs, rewards, terminated, truncated, info = env.step(action)
        
        # TODO: env.render() currently called in step, but should be here instead
        
        if terminated or truncated:
            print(info['episode'])

    env.close()

if __name__ == "__main__":
    main()
