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
from tensorboard.plugins.hparams import api as hp
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
from src.DRL.qnetwork import QNetwork
from src.DRL.wrapped_qrunner import wrapped_qrunner_env

# Based on:
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py#L30

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
    
def sample_hyperparams(hyperparam_ranges):
    hyperparams = {}
    for hyperparam in hyperparam_ranges:
        if isinstance(hyperparam.domain, hp.Discrete):
            hyperparams[hyperparam.name] = random.choice(hyperparam.domain.values)
        elif isinstance(hyperparam.domain, hp.RealInterval):
            hyperparams[hyperparam.name] = round(random.uniform(hyperparam.domain.min_value, hyperparam.domain.max_value), 6)
        elif isinstance(hyperparam.domain, hp.IntInterval):
            hyperparams[hyperparam.name] = random.randint(hyperparam.domain.min_value, hyperparam.domain.max_value)
            
    return hyperparams

if __name__ == "__main__":
    window_size = 10
    num_checkpoints = 5 # + step 0
    log_checkpoints = True
    record_video = False
    human_render = False
    
    hyperparam_ranges = [
        hp.HParam('gamma', hp.RealInterval(0.9, 0.99)),
        hp.HParam('tau', hp.RealInterval(0.9, 1.0)),
        hp.HParam('learning_rate', hp.RealInterval(0.0001, 0.001)),
        hp.HParam('target_network_frequency', hp.IntInterval(500, 2000)),
        hp.HParam('batch_size', hp.Discrete([16, 32, 64, 128])),
        hp.HParam('train_frequency', hp.IntInterval(4, 16)),
        hp.HParam('total_timesteps', hp.Discrete([100_000])),
        hp.HParam('learning_starts', hp.Discrete([1000])),
        hp.HParam('buffer_size', hp.Discrete([100_000])),
        hp.HParam('start_eps', hp.Discrete([1.0])),
        hp.HParam('end_eps', hp.RealInterval(0.01, 0.05)),
        hp.HParam('duration_eps', hp.IntInterval(50_000, 500_000)),
        hp.HParam('frame_skip', hp.Discrete([1, 2, 3, 4])),
        hp.HParam('frame_stack', hp.Discrete([1, 2])),
        hp.HParam('seed', hp.IntInterval(0, 1_000_000)),
    ]
    
    episodic_return_window = WindowMetric(window_size)
    
    hyperparams = sample_hyperparams(hyperparam_ranges)
    print(hyperparams)
    seed = hyperparams['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    task_id = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = str(date)
    model_path = f"runs/{run_name}_task_{task_id}"
    writer = tf.summary.create_file_writer(model_path)
    with writer.as_default():
        hp.hparams(hyperparams)
        
    if log_checkpoints:
        log_increment = math.log10(hyperparams['total_timesteps'])
        log_step_size = log_increment / num_checkpoints
        save_points = [0] + [int(10 ** (log_step_size * i)) for i in range(1, num_checkpoints + 1)]
    else:
        save_points = [int(hyperparams['total_timesteps'] / num_checkpoints) * i for i in range(num_checkpoints + 1)]
    print(f"Saving checkpoints: {save_points}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    q_network = QNetwork(frame_stacks=hyperparams['frame_stack']).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=hyperparams['learning_rate'])
    target_network = QNetwork(frame_stacks=hyperparams['frame_stack']).to(device)
    target_network.load_state_dict(q_network.state_dict())

    ls = LinearSchedule(hyperparams['start_eps'], hyperparams['end_eps'], hyperparams['duration_eps'])
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
    print((rb.observations.nbytes + rb.actions.nbytes + rb.rewards.nbytes + rb.dones.nbytes) / 1e9, "GB", flush=True)
    
    obs, info = env.reset(seed=seed)
    start_time = time.time()
    
    loss = None
    for global_step in range(hyperparams['total_timesteps'] + 1):
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
            episodic_return_window.add(info["episode"]["r"][0])
            with writer.as_default():
                tf.summary.scalar('episodic_return', info["episode"]["r"][0], step=global_step)
                tf.summary.scalar('episodic_length', info["episode"]["l"][0], step=global_step)

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
            if global_step % 1000 == 0 and loss is not None:
                with writer.as_default():
                    tf.summary.scalar('td_loss', loss.item(), step=global_step)
                    tf.summary.scalar('epsilon', epsilon, step=global_step)
                    tf.summary.scalar('q_values', action_q_values.mean().item(), step=global_step)
                    tf.summary.scalar('steps_per_second', int(global_step / (time.time() - start_time)), step=global_step)
            
        # Possibly save model
        if global_step in save_points:
            torch.save(q_network.state_dict(), f"{model_path}/model_{global_step}.pt")
            print(f"Saved checkpoint: {global_step}")
    env.close()
    
    with writer.as_default():
        tf.summary.scalar('final_episodic_return', episodic_return_window.get_mean(), step=global_step)