import csv
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

from src.DRL.qnetwork import QNetwork
from src.DRL.wrapped_qrunner import wrapped_qrunner_env

# DQN algorithm inspired by:
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py#L30

def append_to_csv(file_path, hyperparams, episodic_return):
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as csvfile:
        fieldnames = list(hyperparams.keys()) + ['final_episodic_return']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        data = hyperparams.copy()
        data['final_episodic_return'] = episodic_return
        writer.writerow(data)

class WindowMetric:
    def __init__(self, size):
        self.size = size
        self.window = []

    def add(self, value):
        if len(self.window) == self.size:
            self.window.pop(0)
        self.window.append(value)

    def get_mean(self):
        if not self.window:
            return 0
        return sum(self.window) / len(self.window)

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
    
def realInterval(low, high):
    def _sample():
        return round(random.uniform(low, high), 6)
    return _sample

def intInterval(low, high):
    def _sample():
        return random.randint(low, high)
    return _sample

def discrete(values):
    def _sample():
        return random.choice(values)
    return _sample

def get_default_hyperparams():
    return {
        "gamma": 0.95,
        "tau": 1.0,
        "learning_rate": 0.0001,
        "target_network_frequency": 2000,
        "batch_size": 64,
        "train_frequency": 4,
        "total_timesteps": 5_000_000, # 10m
        "learning_starts": 1000,
        "buffer_size": 500_000,
        "start_eps": 0.5, # 1
        "end_eps": 0.05,
        "duration_eps": 500_000,
        "frame_skip": 5,
    }

if __name__ == "__main__":
    log_interval = 1000
    window_size = 20
    num_checkpoints = 10 # + step 0
    default_hyperparams = True
    time_limit = 60 * 60 * 20 # 2 hours
    
    # Set seeds
    seed = random.randint(0, 1000000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    hyperparam_samplers = {
        'gamma': realInterval(0.9, 0.99),
        'tau': realInterval(0.9, 1.0),
        'learning_rate': realInterval(0.0001, 0.001),
        'target_network_frequency': intInterval(500, 2000),
        'batch_size': discrete([16, 32, 64]),
        'train_frequency': intInterval(4, 16),
        'total_timesteps': discrete([10_000_000]),
        'learning_starts': discrete([1000]), # TODO: make this a factor instead
        'buffer_size': discrete([100_000, 250_000, 500_000]),
        'start_eps': discrete([1.0]),
        'end_eps': realInterval(0.01, 0.05),
        'duration_eps': intInterval(50_000, 500_000),
        'frame_skip': discrete([1, 2, 3, 4, 5]),
    }
    if default_hyperparams:
        hyperparams = get_default_hyperparams()
    else:
        hyperparams = {k: v() for k, v in hyperparam_samplers.items()}
    hyperparams['seed'] = seed
    print(f"Hyperparameters: {hyperparams}")
    
    episodic_return_window = WindowMetric(window_size)
    episodic_length_window = WindowMetric(window_size)
    td_loss_window = WindowMetric(window_size)
    q_values_window = WindowMetric(window_size)
    sps_window = WindowMetric(window_size)
    
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID', '-1') # TODO: simplify name if no slurm
    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = str(date)
    if task_id != '-1':
        model_path = f"runs/{run_name}_task_{task_id}"
    else:
        model_path = f"runs/{run_name}"
    writer = SummaryWriter(model_path)
    writer.add_text("hyperparameters", str(hyperparams))
    
    # save_points = [int(hyperparams['total_timesteps'] / num_checkpoints) * i for i in range(num_checkpoints + 1)]
    save_points = [0, 10**1, 10**1.5, 10**2, 10**2.5, 10**3, 10**3.5, 10**4, 10**4.5, 10**5, 10**5.5, 10**6, 10**6.5, 10**7]
    save_points = [int(x) for x in save_points]
    print(f"Saving at checkpoints: {save_points}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    #q_network = QNetwork().to(device)
    #q_network = QNetwork(model_path="runs/20240416-095130/model_9999000.pt").to(device)
    q_network = QNetwork(model_path="runs/20240417-104259/model_4999000.pt").to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=hyperparams['learning_rate'])
    target_network = QNetwork().to(device)
    target_network.load_state_dict(q_network.state_dict())

    ls = LinearSchedule(hyperparams['start_eps'], hyperparams['end_eps'], hyperparams['duration_eps'])
    env = wrapped_qrunner_env(frame_skip=hyperparams['frame_skip'], original=True)#, human_render=True)
    rb = ReplayBuffer(
        hyperparams['buffer_size'],
        env.observation_space,
        env.action_space,
        device,
        n_envs=1,
        optimize_memory_usage=True,
        handle_timeout_termination=False, # Can't use with optimize_memory_usage
    )
    print(f"Replaybuffer allocates {(rb.observations.nbytes + rb.actions.nbytes + rb.rewards.nbytes + rb.dones.nbytes) / 1e9}GB", flush=True)
    
    obs, info = env.reset(seed=seed)
    start_time = time.time()
    loss = None
    for global_step in range(hyperparams['total_timesteps'] + 1):
        # Determine and perform action
        epsilon = ls.step()
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            tensor_obs = torch.Tensor(obs).to(device)
            reshaped_obs = tensor_obs.unsqueeze(0)
            q_values = q_network(reshaped_obs)
            action = q_values.argmax(dim=1).item()
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Log episodic metrics
        if "episode" in info:
            episodic_return_window.add(info["episode"]["r"][0])
            episodic_length_window.add(info["episode"]["l"])
            writer.add_scalar('episodic_return', episodic_return_window.get_mean(), global_step)
            writer.add_scalar('episodic_length', episodic_length_window.get_mean(), global_step)

        # Handle truncated episodes... TODO...
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
            if global_step % log_interval == 0 and loss is not None:
                td_loss_window.add(loss.item())
                q_values_window.add(action_q_values.mean().item())
                writer.add_scalar('td_loss', td_loss_window.get_mean(), global_step)
                writer.add_scalar('q_values', q_values_window.get_mean(), global_step)
                
                writer.add_scalar('epsilon', epsilon, global_step)
                writer.add_scalar('steps_per_second', int(global_step / (time.time() - start_time)), global_step)
        
        time_done = time.time() - start_time > time_limit    
        
        # Possibly save model
        steps_after_start = global_step - hyperparams['learning_starts']
        if steps_after_start in save_points or time_done or global_step >= hyperparams['total_timesteps']:
            torch.save(q_network.state_dict(), f"{model_path}/model_{steps_after_start}.pt")
            print(f"Saved checkpoint: {steps_after_start}")
            
        if time_done:
            print("Time limit reached")
            break
            
    final_episodic_return_mean = episodic_return_window.get_mean()
    results_file_path = f"runs/results_summary.csv"
    append_to_csv(results_file_path, hyperparams, final_episodic_return_mean)
    print(f"Appended final results to {results_file_path}")
    
    env.close()