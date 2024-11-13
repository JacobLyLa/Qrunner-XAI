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
        "learning_rate": 0.00005,
        "gamma": 0.95,
        "batch_size": 64,
        "train_frequency": 5,
        "target_model_frequency": 2000,
        "total_timesteps": 10_000_000,
        "learning_starts_fraction": 0.01,
        "duration_eps_fraction": 0.3,
        "buffer_size": 500_000,
        "start_eps": 1.0,
        "end_eps": 0.01,
        "frame_skip": 4,
        "max_steps": 2000,
        "max_steps_reward": 150,
        "blending_alpha": 0.7,
        "grad_clip": 1.0,
        "use_grayscale": False,
        "use_double": False,
        "use_dueling": False,
    }

def train(hyperparams):
    log_interval = 1000
    time_limit = 60 * 60 * 2 # 2 hour
    eval_episodes = 100
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(f"Hyperparameters: {hyperparams}")
    
    # Calculate actual learning_starts and duration_eps
    learning_starts = int(hyperparams['learning_starts_fraction'] * hyperparams['total_timesteps'])
    duration_eps = int(hyperparams['duration_eps_fraction'] * hyperparams['total_timesteps'])

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
    last_log_time = start_time

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

        if not truncated:
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
                
                # Add l2 regularization
                l2_loss = sum(torch.norm(p) for p in model.parameters())
                loss += 0.00001 * l2_loss

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparams['grad_clip'])
                
                optimizer.step()
            
            # Update target network
            if global_step % hyperparams['target_model_frequency'] == 0:
                target_model.load_state_dict(model.state_dict())

            # Log non-episodic metrics
            if global_step % log_interval == 0:
                current_time = time.time()
                elapsed_time = current_time - last_log_time
                fps = log_interval / elapsed_time if elapsed_time > 0 else 0.0
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
        time_done = time.time() - start_time > time_limit
        
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
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = model(torch.Tensor(obs).unsqueeze(0).to(device)).argmax(dim=1).item()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        eval_rewards.append(episode_reward)

    avg_eval_reward = sum(eval_rewards) / len(eval_rewards)
    print(f"Average evaluation reward: {avg_eval_reward}")

    # Log evaluation results
    wandb.log({"eval/average_reward": avg_eval_reward})
    
    # Save model
    torch.save(model.state_dict(), f"models/qrunner_dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")

    env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="Run as part of a wandb sweep")
    args = parser.parse_args()

    if args.sweep:
        wandb.init()
    else:
        wandb.init(
            project="Qrunner-XAI",
            entity="jacob-llarsen",
            name=str(datetime.now().strftime("%Y%m%d-%H%M%S")),
            config=get_default_hyperparams()
        )
        
    train(wandb.config)

    wandb.finish()