import os
import random
import time
from datetime import datetime

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from custom_env import make_env


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 4), # 
            #nn.Linear(512, 32),
            #nn.ReLU(),
            #nn.Linear(32, 4),
        )

    def forward(self, x):
        return self.network(x / 255.0) # TODO: needed? if trained like this, then CAV needs to use this aswell!
        # return self.network(x.float())

def load_model(model_path):
    model = QNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

if __name__ == "__main__":
    # TODO: cleanup
    seed = 1
    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    buffer_size = 1_000_000
    gamma = 0.99
    tau = 1.0
    learning_rate = 0.0001
    total_timesteps = 10_000_000
    start_e = 1.0
    end_e = 0.01
    exploration_fraction = 0.1
    target_network_frequency = 1000
    learning_starts = 100_000
    train_frequency = 4
    batch_size = 32
    save_model = True
    run_name = f"dqn_breakout/{date}"

    # create folders
    os.makedirs(f"runs/{run_name}/models", exist_ok=True)
    os.makedirs(f"runs/{run_name}/images", exist_ok=True)

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        f"buffer_size: {buffer_size}\n"
        f"gamma: {gamma}\n"
        f"tau: {tau}\n"
        f"learning_rate: {learning_rate}\n"
        f"seed: {seed}\n"
        f"total_timesteps: {total_timesteps}\n"
        f"start_e: {start_e}\n"
        f"end_e: {end_e}\n"
        f"exploration_fraction: {exploration_fraction}\n"
        f"target_network_frequency: {target_network_frequency}\n"
        f"learning_starts: {learning_starts}\n"
        f"train_frequency: {train_frequency}\n"
        f"batch_size: {batch_size}\n"
        f"run_name: {run_name}\n"
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    env = make_env(seed, run_name)

    q_network = QNetwork().to(device)
    #q_network = load_model("runs/dqn_breakout/20230824-180025/models/model_9900000.pt").to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    target_network = QNetwork().to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        buffer_size,
        env.observation_space,
        env.action_space,
        device,
        n_envs=1,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    obs, info = env.reset(seed=seed)
    for global_step in range(total_timesteps):
        # action logic
        epsilon = linear_schedule(start_e, end_e, exploration_fraction * total_timesteps, global_step)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            np_obs = np.array(obs)
            tensor_obs = torch.Tensor(np_obs).to(device)
            reshaped_obs = tensor_obs.unsqueeze(0)
            q_values = q_network(reshaped_obs)
            action = q_values.argmax(dim=1).item()

        next_obs, reward, terminated, truncated, info = env.step(action)

        # handle `final_observation'
        real_next_obs = next_obs
        if terminated:
            real_next_obs = info["final_observation"]
            writer.add_scalar("charts/episodic_return", info["final_info"]["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", info["final_info"]["episode"]["l"], global_step)
            writer.add_scalar("charts/epsilon", epsilon, global_step)
        action = np.array([action]) # rb expects numpy array
        rb.add(obs, real_next_obs, action, reward, terminated, info)

        # important
        obs = next_obs

        if global_step > learning_starts:
            if global_step % train_frequency == 0:
                data = rb.sample(batch_size)
                # TODO: simplify or understand better
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if global_step % 1000 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                sps = int(global_step / (time.time() - start_time))
                writer.add_scalar("charts/SPS", sps, global_step)

            # update target network
            if global_step % target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        tau * q_network_param.data + (1.0 - tau) * target_network_param.data
                    )

        if save_model and global_step > learning_starts:
            if global_step % (total_timesteps // 20) == 0 or global_step == total_timesteps - 1:
                model_path = f"runs/{run_name}/models/model_{global_step}.pt"
                torch.save(q_network.state_dict(), model_path)
                print(f"model saved to {model_path}")
    env.close()