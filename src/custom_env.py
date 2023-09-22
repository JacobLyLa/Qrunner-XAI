import os

import gymnasium as gym
from stable_baselines3.common.atari_wrappers import ClipRewardEnv

from utils import prepare_folders


def make_env(seed, state_observer=None, render_mode="rgb_array", record_video=False):
    env = gym.make("BreakoutNoFrameskip-v4", render_mode=render_mode)
    if record_video:
        prepare_folders(f"../runs/videos")
        env = gym.wrappers.RecordVideo(env, f"../runs/videos")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.AtariPreprocessing(
        env = env,
        noop_max = 30,
        frame_skip = 4,
        screen_size = 84,
        terminal_on_life_loss = False,
        grayscale_obs = True,
        grayscale_newaxis = False,
        scale_obs = False,
    )
    env = ClipRewardEnv(env) # TODO: needed?
    # TODO max frames for agent
    env = gym.wrappers.FrameStack(env, 4)
    env = gym.wrappers.AutoResetWrapper(env)
    if state_observer is not None:
        env = state_observer(env)

    env.action_space.seed(seed)

    return env