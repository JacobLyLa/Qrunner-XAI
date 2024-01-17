import random

from wrapped_q_runner import make_env

def main():
    env = make_env(render_mode='human')
    obs, info = env.reset()
    total_reward = 0
    total_episodes = 0
    for _ in range(5000):
        action = 2 if random.random() < 0.9 else 3
        obs, rewards, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(info['episode'])
            total_reward += info['episode']['r']
            total_episodes += 1
            obs, info = env.reset()
    env.close()
    print(f"Average reward: {total_reward / total_episodes}")

if __name__ == "__main__":
    main()
