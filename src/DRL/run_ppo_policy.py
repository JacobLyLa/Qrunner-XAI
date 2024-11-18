import os
import torch
import numpy as np

from src.DRL.ppo_network import ActorCriticNetwork
from src.DRL.wrapped_qrunner import wrapped_qrunner_env
from torch.distributions import Categorical

def run_ppo_policy(model_path, env, device, max_episodes=1000):
    # Load the trained policy
    policy = ActorCriticNetwork(colors=env.observation_space.shape[-1], actions=env.action_space.n).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()
    
    episode = 0
    episode_rewards = []
    max_episodes = max_episodes
    
    total_episodes = 0
    total_rewards = 0
    max_reward = float('-inf')
    
    print(f"Starting PPO Policy Run for {max_episodes} episodes")
    
    while episode < max_episodes:
        state, info = env.reset()
        state = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(device)
        done = False
        episode_reward = 0
        
        while not done:            
            with torch.no_grad():
                action_logits, _ = policy(state)
                dist = Categorical(logits=action_logits)
                action = dist.sample().item()
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            state = torch.FloatTensor(next_state).permute(2, 0, 1).unsqueeze(0).to(device)
        
        episode += 1
        total_episodes += 1
        total_rewards += episode_reward
        max_reward = max(max_reward, episode_reward)
        episode_rewards.append(episode_reward)
        
        if episode % 10 == 0 or episode == max_episodes:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode}/{max_episodes} - Average Reward: {avg_reward:.2f}")
    
    print("PPO Policy Run Complete")
    print(f"Total Episodes: {total_episodes}")
    print(f"Average Reward: {total_rewards / total_episodes:.2f}")
    print(f"Max Reward: {max_reward:.2f}")
    env.close()

def main():
    model_path = "runs/ppo_20241118-155434/ppo_model_final.pt"  # Update with your model path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = wrapped_qrunner_env(frame_skip=4, original=True, human_render=True)
    
    run_ppo_policy(model_path=model_path, env=env, device=device, max_episodes=1000)

if __name__ == "__main__":
    main()