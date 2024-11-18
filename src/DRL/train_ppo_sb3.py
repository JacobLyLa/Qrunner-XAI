import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from src.DRL.wrapped_qrunner import wrapped_qrunner_env

def main():
    # Define environment parameters
    frame_skip = 4
    human_render = False
    auto_reset = False

    # Create the environment
    env = wrapped_qrunner_env(
        frame_skip=frame_skip,
        human_render=human_render,
        auto_reset=auto_reset,
    )

    # Define the PPO model
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=2.5e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        tensorboard_log="./ppo_sb3_tensorboard/"
    )

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./models/',
        name_prefix='ppo_sb3_model'
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path='./models/best_model/',
        log_path='./logs/',
        eval_freq=5000,
        deterministic=False,
        render=False
    )

    # Create directories if they don't exist
    os.makedirs('./models/', exist_ok=True)
    os.makedirs('./models/best_model/', exist_ok=True)
    os.makedirs('./logs/', exist_ok=True)
    os.makedirs('./ppo_sb3_tensorboard/', exist_ok=True)

    # Train the model
    model.learn(
        total_timesteps=200000,
        callback=[checkpoint_callback, eval_callback]
    )

    # Save the final model
    model.save("models/ppo_sb3_final")

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()