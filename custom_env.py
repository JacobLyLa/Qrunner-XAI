import os

import gymnasium as gym
from stable_baselines3.common.atari_wrappers import ClipRewardEnv

from state_observer import StateObserverWrapper
from utils import prepare_folders


def make_env(seed, run_name, state_observer=False, render_mode="rgb_array", record_video=False):
    env = gym.make("BreakoutNoFrameskip-v4", render_mode=render_mode)
    if record_video:
        prepare_folders(f"runs/{run_name}/videos")
        env = gym.wrappers.RecordVideo(env, f"runs/{run_name}/videos")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.AtariPreprocessing(
        env = env,
        noop_max = 30,
        frame_skip = 4,
        screen_size = 84,
        terminal_on_life_loss = False,
        grayscale_obs = True,
        grayscale_newaxis = False,
        scale_obs = False, # TODO: this with True requires much more VRAM.
        # just divide by 255.0 in the network instead?
    )
    env = ClipRewardEnv(env) # TODO: why isnt this in atari preprocessing?
    env = gym.wrappers.FrameStack(env, 4)
    env = gym.wrappers.AutoResetWrapper(env) # TODO: needed?
    if state_observer:
        env = StateObserverWrapper(env)
    env.action_space.seed(seed)

    return env