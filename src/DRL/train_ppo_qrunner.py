import os
import time
import random
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn.functional import mse_loss

from src.DRL.ppo_network import ActorCriticNetwork
from src.DRL.wrapped_qrunner import wrapped_qrunner_env
from torch.utils.tensorboard import SummaryWriter

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.95, K_epochs=4, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCriticNetwork(colors=state_dim[-1], actions=action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.policy_old = ActorCriticNetwork(colors=state_dim[-1], actions=action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = mse_loss

    def select_action(self, state):
        with torch.no_grad():
            action_logits, state_value = self.policy_old(state)
            dist = Categorical(logits=action_logits)
            action = dist.sample()
            return action.item(), dist.log_prob(action), state_value

    def update(self, memory):
        # Convert lists to tensors
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalize rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            action_logits, state_values = self.policy(old_states)
            dist = Categorical(logits=action_logits)
            
            # New log probs
            logprobs = dist.log_prob(old_actions)
            entropy = dist.entropy()
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Advantages
            # advantages = rewards - state_values.detach().squeeze()
            # Implement GAE
            gaes = []
            gae = 0
            for reward, is_terminal, value in zip(reversed(memory.rewards), reversed(memory.is_terminals), reversed(state_values)):
                delta = reward + self.gamma * value * (1 - is_terminal) - value
                gae = delta + self.gamma * self.gamma * gae
                gaes.insert(0, gae)
            advantages = torch.tensor(gaes, dtype=torch.float32).to(device)

            # Surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values.squeeze(), rewards) - 0.01 * entropy

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

if __name__ == "__main__":
    log_interval = 10
    save_interval = 100
    max_episodes = 1000
    lr = 1e-4
    gamma = 0.95
    K_epochs = 4
    eps_clip = 0.2
    update_timestep = 1000  # Total timesteps before updating the policy

    # Set seeds
    seed = random.randint(0, 1000000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize environment
    env = wrapped_qrunner_env(frame_skip=4, original=True)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n

    # Initialize agent
    agent = PPOAgent(state_dim, action_dim, lr, gamma, K_epochs, eps_clip)
    memory = Memory()

    # Logging
    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"ppo_{date}"
    model_path = f"runs/{run_name}"
    os.makedirs(model_path, exist_ok=True)
    writer = SummaryWriter(model_path)

    # Training loop
    timestep = 0
    episode_rewards = []
    for episode in range(1, max_episodes + 1):
        state, _ = env.reset(seed=seed)
        state = torch.FloatTensor(state).permute(2, 0, 1).to(device)
        episode_reward = 0
        done = False
        truncated = False

        while not done and not truncated:
            timestep +=1

            # Select action
            action, logprob, value = agent.select_action(state.unsqueeze(0))
            next_state, reward, done, truncated, info = env.step(action)
            episode_reward += reward

            # Save to memory
            memory.states.append(state)
            memory.actions.append(torch.tensor(action).to(device))
            memory.logprobs.append(logprob)
            memory.rewards.append(reward)
            memory.is_terminals.append(done or truncated)

            state = torch.FloatTensor(next_state).permute(2, 0, 1).to(device)

            # Update PPO
            if timestep % update_timestep == 0:
                agent.update(memory)
                memory.clear_memory()

        episode_rewards.append(episode_reward)
        writer.add_scalar('Reward/Episode', episode_reward, episode)

        # Logging
        if episode % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            print(f"Episode {episode} | Step {timestep} | Average Reward: {avg_reward:.2f}")
            writer.add_scalar('Reward/Average', avg_reward, episode)

        # Save model
        if episode % save_interval == 0:
            torch.save(agent.policy.state_dict(), f"{model_path}/ppo_model_episode_{episode}.pt")

    # Save final model
    torch.save(agent.policy.state_dict(), f"{model_path}/ppo_model_final.pt")
    print("Training complete. Model saved.")

    env.close()