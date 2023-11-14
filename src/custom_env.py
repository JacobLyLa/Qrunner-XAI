import os

import gymnasium as gym
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, FireResetEnv

from game_step_saver import GameStepSaverWrapper
from utils import prepare_folders
        
### if save_interval is 0, no states is saved/tracked
def make_env(save_interval=0, render_mode="rgb_array", record_video=False):
    env = gym.make("BreakoutNoFrameskip-v4", render_mode=render_mode)
    if record_video:
        prepare_folders(f"../runs/videos")
        env = gym.wrappers.RecordVideo(env, f"../runs/videos")
    env = gym.wrappers.AutoResetWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.AtariPreprocessing(
        env = env,
        noop_max = 30,
        frame_skip = 4,
        screen_size = 84,
        terminal_on_life_loss = True,
        grayscale_obs = True,
        grayscale_newaxis = False,
        scale_obs = False,
    )
    env = ClipRewardEnv(env)
    env = gym.wrappers.FrameStack(env, 4)
    if save_interval:
        env = GameStepSaverWrapper(env, save_interval=save_interval)
    return env