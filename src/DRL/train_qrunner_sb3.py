import time

from stable_baselines3 import DQN
from src.DRL.wrapped_qrunner import wrapped_qrunner_env


def create_or_load_model(algorithm, env, restart, model_name, tensorboard_log):
    if restart:
        if algorithm is DQN:
            print("Creating new DQN model")
            # Might want to set stat_windows to 1 instead of 100
            model = algorithm("CnnPolicy", env, verbose=0, tensorboard_log=tensorboard_log, buffer_size=100000, gamma=0.98)
        else:
            print("Creating new PPO model")
            model = algorithm("CnnPolicy", env, verbose=0, tensorboard_log=tensorboard_log)
    else:
        model = algorithm.load(model_name)
        model.set_env(env)
    return model

def train_model(use_dqn, env, its, restart=True):
    algorithm = DQN if use_dqn else PPO
    tensorboard_log = "runs/sb3/"
    model_name = f"{tensorboard_log}/dqn_qrunner" if use_dqn else f"{tensorboard_log}/ppo_qrunner"

    model = create_or_load_model(algorithm, env, restart, model_name, tensorboard_log)

    for i in range(10):
        reset_timesteps = True if i == 0 and restart else False
        model.learn(total_timesteps=its//10, reset_num_timesteps=reset_timesteps, progress_bar=True)
        model.save(model_name)
        print(f"[{i}] Model saved as {model_name}")

    return model

def main():
    env = wrapped_qrunner_env(frame_skip=4)
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    
    #model = train_model(True, env, 1000000, True)
    #model = DQN.load("runs/sb3/dqn_qrunner")

    # Test the trained agent
    env = wrapped_qrunner_env(frame_skip=4, human_render=True, scale=6)
    model.set_env(env)
    obs, info = env.reset()
    last_time = time.time()
    target_fps = 50
    for _ in range(10000):
        current_time = time.time()
        time.sleep(max(0, 1/target_fps - (current_time - last_time)))
        last_time = current_time
        
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(info['episode'])
    env.close()

if __name__ == "__main__":
    main()
