import itertools
import math
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.DRL.qnetwork import QNetwork
from src.DRL.wrapped_qrunner import wrapped_qrunner_env

# Inspired by:
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py#L30


class LinearSchedule:
    def __init__(self, start, end, duration):
        self.start = start
        self.end = end
        self.duration = duration
        self.step_count = 0

    def step(self):
        if self.step_count >= self.duration:
            return self.end

        value = self.start + (self.end - self.start) * (self.step_count / self.duration)
        self.step_count += 1
        return value

def random_hyperparam_sample(hyperparam_ranges):
    sample = {}
    for param, values in hyperparam_ranges.items():
        discrete = values[-1]
        values = values[:-1]
        if discrete:
            sample[param] = random.choice(values)
        elif len(values) == 2:
            if all(isinstance(v, float) for v in values):
                sample[param] = random.uniform(values[0], values[1])
            else:
                sample[param] = random.randint(values[0], values[1])
    return sample

# Ints -> Discrete sample
# 2 Floats -> Uniform sample
# Last bool: discrete (choice)
hyperparam_ranges = {
    "gamma": (0.9, 0.99, False),
    "tau": (0.9, 1.0, False),
    "learning_rate": (0.00001, 0.001, False),
    "target_network_frequency": (500, 2000, False),
    "batch_size": (16, 32, 64, 128, 256, True),
    "train_frequency": (2, 32, False),
    "total_timesteps": (100_000, True),
    "learning_starts": (1000, 50000, False),
    "buffer_size": (100_000, True),
    "start_eps": (1.0, True),
    "end_eps": (0.01, 0.05, False),
    "duration_eps": (5000, 200_000, False),
    "frame_skip": (2, 8, False),
    "frame_stack": (2, 4, False),
}

if __name__ == "__main__":
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    seed = random.randint(0, 1000000) # Saved for reproduction
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    num_checkpoints = 10 # + step 0
    log_checkpoints = False
    record_video = False
    human_render = False
    
    hyperparams = random_hyperparam_sample(hyperparam_ranges)
    hyperparams['seed'] = seed
    print(f"Using hyperparams:")
    print(hyperparams)

    if log_checkpoints:
        log_increment = math.log10(hyperparams['total_timesteps'])
        log_step_size = log_increment / num_checkpoints
        save_points = [0] + [int(10 ** (log_step_size * i)) for i in range(1, num_checkpoints + 1)]
    else:
        save_points = [int(hyperparams['total_timesteps'] / num_checkpoints) * i for i in range(num_checkpoints + 1)]
    print(f"Saving checkpoints: {save_points}")

    ls = LinearSchedule(hyperparams['start_eps'], hyperparams['end_eps'], hyperparams['duration_eps'])

    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = str(date)
    model_path = f"runs/{run_name}_task_{task_id}"

    writer = SummaryWriter(model_path)
    writer.add_text("hyperparameters", str(hyperparams))

    q_network = QNetwork(frame_stacks=hyperparams['frame_stack']).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=hyperparams['learning_rate'])
    target_network = QNetwork(frame_stacks=hyperparams['frame_stack']).to(device)
    target_network.load_state_dict(q_network.state_dict())

    env = wrapped_qrunner_env(frame_skip=hyperparams['frame_skip'], frame_stack=hyperparams['frame_stack'], human_render=human_render, record_video=record_video)
    rb = ReplayBuffer(
        hyperparams['buffer_size'],
        env.observation_space,
        env.action_space,
        device,
        n_envs=1,
        optimize_memory_usage=True,
        handle_timeout_termination=False, # Can't use with optimize_memory_usage
    )
    obs, info = env.reset(seed=seed)

    start_time = time.time()
    for global_step in tqdm(range(hyperparams['total_timesteps'])):
        # Determine and perform action
        epsilon = ls.step()
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            np_obs = np.array(obs)
            tensor_obs = torch.Tensor(np_obs).to(device)
            reshaped_obs = tensor_obs.unsqueeze(0)
            q_values = q_network(reshaped_obs)
            action = q_values.argmax(dim=1).item()
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Record end of episode stats
        if "episode" in info:
            # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
            writer.add_scalar("charts/epsilon", epsilon, global_step)

        # Handle truncated episodes
        real_next_obs = next_obs
        if truncated:
            real_next_obs = info["final_observation"]
        rb.add(obs, real_next_obs, np.array([action]), reward, terminated, info)
        obs = next_obs

        # Possibly train
        if global_step >= hyperparams['learning_starts']:
            if global_step % hyperparams['train_frequency'] == 0:
                data = rb.sample(hyperparams['batch_size'])
                with torch.no_grad():
                    next_observation_q_values = target_network(data.next_observations)
                    # Extract maximum Q-value for each next observation.
                    target_max_q_values, _ = next_observation_q_values.max(dim=1)
                    
                    # Calculate the TD target for each observation.
                    # If episode ends (done flag is 1), the future Q-value is not considered.
                    future_values = hyperparams['gamma'] * target_max_q_values * (1 - data.dones.flatten())
                    td_targets = data.rewards.flatten() + future_values

                # Compute Q-values of current observations using the main Q-network.
                current_observation_q_values = q_network(data.observations)

                # Extract the Q-values of the taken actions.
                action_q_values = current_observation_q_values.gather(1, data.actions).squeeze()

                # Compute the mean squared error loss.
                loss = mse_loss(td_targets, action_q_values)

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update target network
            if global_step % hyperparams['target_network_frequency'] == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        hyperparams['tau'] * q_network_param.data + (1.0 - hyperparams['tau']) * target_network_param.data
                    )

            # Log non-episodic metrics
            if global_step % 1000 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", action_q_values.mean(), global_step)
                sps = int(global_step / (time.time() - start_time))
                writer.add_scalar("charts/SPS", sps, global_step)
            
        # Possibly save model
        if global_step in save_points:
            torch.save(q_network.state_dict(), f"{model_path}/model_{global_step}.pt")
            print(f"Saved checkpoint: {global_step}")
    env.close()