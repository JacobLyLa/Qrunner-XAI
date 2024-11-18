import os
from stable_baselines3 import PPO
from src.DRL.wrapped_qrunner import wrapped_qrunner_env

def main():
    # Define environment parameters
    frame_skip = 4
    human_render = True

    # Create the environment
    env = wrapped_qrunner_env(
        frame_skip=frame_skip,
        human_render=human_render,
    )

    # Load the trained PPO model
    model = PPO.load("models/ppo_sb3_final", env=env)

    # Run the model
    episodes = 10
    obs, _ = env.reset()
    done = False
    for _ in range(episodes):
        while not done:
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            # done = terminated or truncated
    env.close()

if __name__ == "__main__":
    main()