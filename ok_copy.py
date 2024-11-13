import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.nn.functional import mse_loss
from src.Qrunner.qrunner import QrunnerEnv
from src.DRL.DQN import DQN
from src.DRL.wrapped_qrunner import QrunnerWrapper, HumanRenderWrapper
import wandb

def linear_schedule(start, end, duration, step):
    if step >= duration:
        return end
    return start + (end - start) * (step / duration)
    
def get_default_hyperparams():
    return {
        "gamma": 0.95,
        "learning_rate": 0.0001,
        "target_model_frequency": 2000,
        "batch_size": 64,
        "train_frequency": 4,
        "total_timesteps": 1_000_000,
        "learning_starts_fraction": 0.01,
        "buffer_size": 500_000,
        "start_eps": 0.8,
        "end_eps": 0.05,
        "duration_eps_fraction": 0.3,
        "frame_skip": 4,
        "max_steps": 1000,
        "max_steps_reward": 100,
        "blending_alpha": 0.8,
        "use_grayscale": False,
        "use_double": True,
        "use_dueling": True,
    }

if __name__ == "__main__":
    run_name = str(datetime.now().strftime("%Y%m%d-%H%M%S"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    hyperparams = get_default_hyperparams()
    log_interval = 1000
    time_limit = 60 * 60 * 1 # 1 hour
    eval_episodes = 100
    print(f"Hyperparameters: {hyperparams}")
    
    # Calculate actual learning_starts and duration_eps
    learning_starts = int(hyperparams['learning_starts_fraction'] * hyperparams['total_timesteps'])
    duration_eps = int(hyperparams['duration_eps_fraction'] * hyperparams['total_timesteps'])

    # Initialize wandb
    wandb.init(
        project="Qrunner-DQN",
        entity="jacob-llarsen",
        name=run_name,
        config=hyperparams
    )
    
    # Initialize environment with grayscale option
    env = QrunnerWrapper(
        QrunnerEnv(),
        max_steps=hyperparams['max_steps'],
        max_steps_reward=hyperparams['max_steps_reward'],
        blending_alpha=hyperparams['blending_alpha'],
        frame_skip=hyperparams['frame_skip'],
        use_grayscale=hyperparams['use_grayscale']
    )
    #env = HumanRenderWrapper(env, scale=6, fps=0, wrapped_render=True)

    # Initialize DQN with appropriate input channels and hyperparameters
    input_channels = 1 if hyperparams['use_grayscale'] else 3
    model = DQN(input_channels=input_channels, use_dueling=hyperparams['use_dueling']).to(device)
    target_model = DQN(input_channels=input_channels, use_dueling=hyperparams['use_dueling']).to(device)
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
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
    
    obs, info = env.reset()
    done = False
    start_time = time.time()
    
    # Initialize FPS tracking
    last_log_time = time.time()

    for global_step in range(hyperparams['total_timesteps'] + 1):
        if done:
            obs, info = env.reset()
        # Determine and perform action
        epsilon = linear_schedule(hyperparams['start_eps'], hyperparams['end_eps'], duration_eps, global_step)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = model(torch.Tensor(obs).unsqueeze(0).to(device)).argmax(dim=1).item()
        next_obs, reward, terminated, truncated, info = env.step(action)

        if truncated:
            rb.add(obs, obs, np.array([action]), reward, terminated, info)
        else:
            rb.add(obs, next_obs, np.array([action]), reward, terminated, info)
        obs = next_obs

        # Possibly train
        if global_step >= learning_starts:
            if global_step % hyperparams['train_frequency'] == 0:
                data = rb.sample(hyperparams['batch_size'])
                
                with torch.no_grad():
                    if hyperparams['use_double']:
                        # Double DQN
                        actions_next = model(data.next_observations).argmax(dim=1, keepdim=True)
                        next_q_values = target_model(data.next_observations)
                        target_max_q_values = next_q_values.gather(1, actions_next).squeeze()
                    else:
                        # Standard DQN
                        target_max_q_values = target_model(data.next_observations).max(dim=1)[0]
                    
                    # Compute TD targets
                    td_targets = data.rewards.flatten() + hyperparams['gamma'] * target_max_q_values * (1 - data.dones.flatten())

                # Compute current Q values
                current_values = model(data.observations).gather(1, data.actions).squeeze()
                
                # Calculate loss
                loss = mse_loss(td_targets, current_values)

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Update target network
            if global_step % hyperparams['target_model_frequency'] == 0:
                target_model.load_state_dict(model.state_dict())

            # Log non-episodic metrics
            if global_step % hyperparams['log_interval'] == 0:
                current_time = time.time()
                elapsed_time = current_time - last_log_time
                fps = hyperparams['log_interval'] / elapsed_time if elapsed_time > 0 else 0.0
                last_log_time = current_time

                wandb.log({
                    "TD Loss": loss.item(),
                    "Epsilon": epsilon,
                    "Average Q-value": current_values.mean().item(),
                    "FPS": fps  # Logging FPS
                }, step=global_step)
        
        # Log episodic metrics
        done = terminated or truncated
        if done:
            # Count occurrences of each interaction type
            interactions = info['interactions']
            interaction_counts = {
                "(E) Gold coin count": interactions.count("gold coin"),
                "(E) Red coin count": interactions.count("red coin"),
                "(E) Blue coin count": interactions.count("blue coin"),
                "(E) Bullet count": interactions.count("bullet"),
                "(E) Lava count": interactions.count("lava"),
            }

            # Convert truncated to 1 if True, else 0
            truncated_flag = 1.0 if truncated else 0.0

            # Log the metrics with individual interaction counts
            wandb.log({
                "(E) Total reward": info['total_reward'],
                "(E) DQN step": info['wrapper_steps'],
                "(E) Level progress": info['level_progress'],
                **interaction_counts,
                "(E) Truncated": truncated_flag
            }, step=global_step)
        time_done = time.time() - start_time > hyperparams['time_limit']
        
        if time_done:
            print("Time limit reached")
            break
            
    # Evaluation loop
    print("Starting evaluation...")
    eval_rewards = []
    for _ in range(eval_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        while not done:
            with torch.no_grad():
                action = model(torch.Tensor(obs).unsqueeze(0).to(device)).argmax(dim=1).item()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        eval_rewards.append(episode_reward)

    avg_eval_reward = sum(eval_rewards) / len(eval_rewards)
    print(f"Average evaluation reward: {avg_eval_reward}")

    # Log evaluation results
    wandb.log({
        "eval/average_reward": avg_eval_reward,
        "eval/min_reward": min(eval_rewards),
        "eval/max_reward": max(eval_rewards),
    })

    env.close()
    
    # Finish wandb run
    wandb.finish()