import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from gymnasium.wrappers import FrameStack, ResizeObservation
from stable_baselines3.common.atari_wrappers import (ClipRewardEnv,
                                                     MaxAndSkipEnv,
                                                     NoopResetEnv, WarpFrame)

from src.Qrunner.qrunner import QrunnerEnv


# custom wrapper that combines the 4 stacked frames and colors to same channel so the result is (batch_size, 4*3, 84, 84)
class Reshape(gym.ObservationWrapper):
    def __init__(self, env, screen_size, frame_stack, color_channels):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, 
                                                shape=(frame_stack*color_channels, screen_size, screen_size), 
                                                dtype=np.uint8)

    def observation(self, observation):
        # Convert LazyFrames to numpy array. Probably loses performance
        observation = np.array(observation)

        # Assuming the observation is a numpy array with shape (num_stacks, height, width, channels)
        num_stacks, height, width, channels = observation.shape

        # Reshape to combine stacks and channels
        reshaped_obs = observation.transpose(1, 2, 0, 3).reshape(height, width, -1)

        # Transpose to (channels*stacks, height, width)
        reshaped_obs = reshaped_obs.transpose(2, 0, 1)

        return reshaped_obs


# TODO: one giant custom wrapper with everything combined (maybe not framestack though)
def wrapped_qrunner_env(size, render_mode="rgb_array", record_video=False, frame_skip=2, frame_stack=2):
    final_size = 84
    env = QrunnerEnv(render_mode=render_mode, size=size)

    if record_video:
        env = gym.wrappers.RecordVideo(env, f"../runs/videos")
    
    if frame_skip > 1:
        env = MaxAndSkipEnv(env, skip=frame_skip)  # Skips frames and repeats action and sums reward. max not needed
    #env = WarpFrame(env, width=size, height=size)  # Resize and grayscale
    if size != final_size:
        print("Resizing observation to 84x84")
        env = ResizeObservation(env, shape=(final_size, final_size))
    print("Obs space:", env.observation_space.shape)
    env = FrameStack(env, frame_stack)  # Stack frames, SB3 doesn't allow cnn policy with obs: (x, 42, 42, 3)
    #env = ClipRewardEnv(env) # Can test, but then walking right and collecting coin etc is equal?
    env = Reshape(env, final_size, frame_stack, 3)
    print("Obs space after stacking:", env.observation_space.shape)

    # Additional wrappers for monitoring and auto-reset
    env = gym.wrappers.AutoResetWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    return env

def main():
    import random

    import matplotlib.pyplot as plt
    from stable_baselines3.common.env_checker import check_env
    from tqdm import tqdm
    
    env = wrapped_qrunner_env(size=84*4, frame_skip=4, frame_stack=2, render_mode='rgb_array', record_video=False)
    check_env(env, warn=True)
    obs = env.reset()
    num_frames = 100
    save_path = 'figures/observations/'

    for i in tqdm(range(num_frames), desc="Processing frames"):
        action = 3 if random.random() < 0.2 else 2 # move right and sometimes jump
        obs, rewards, terminated, truncated, info = env.step(action)
        obs = np.concatenate(obs, axis=1)
        # Save the observation as an image
        if i % 5 == 0:
            plt.imshow(obs, cmap='gray')
            plt.title(f"Frame {i}")
            plt.savefig(f"{save_path}frame_{i}.png")
            plt.close()

    print(f"Frames saved to {save_path}")

if __name__ == "__main__":
    main()