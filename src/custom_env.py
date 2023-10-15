import os

import gymnasium as gym
from stable_baselines3.common.atari_wrappers import ClipRewardEnv

from game_step_saver import GameStepSaverWrapper
from utils import prepare_folders


# Overrides action in step if lives just changed to fire
class AutoFireWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = None

    def step(self, action):
        if self.lives != self.env.unwrapped.ale.lives():
            action = 1
            self.lives = self.env.unwrapped.ale.lives()
        return self.env.step(action)

### if save_interval is 0, no states is saved/tracked
def make_env(seed, save_interval=0, render_mode="rgb_array", record_video=False):
    env = gym.make("BreakoutNoFrameskip-v4", render_mode=render_mode)
    env.action_space.seed(seed)
    if record_video:
        prepare_folders(f"../runs/videos")
        env = gym.wrappers.RecordVideo(env, f"../runs/videos")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.AtariPreprocessing(
        env = env,
        noop_max = 30,
        frame_skip = 4,
        screen_size = 84,
        terminal_on_life_loss = False, # TODO: check if true improves performance
        grayscale_obs = True,
        grayscale_newaxis = False,
        scale_obs = False,
    )
    env = ClipRewardEnv(env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=5000)
    env = gym.wrappers.FrameStack(env, 4)
    if save_interval:
        env = GameStepSaverWrapper(env, save_interval=save_interval)
    env = AutoFireWrapper(env) # needs to be second last
    env = gym.wrappers.AutoResetWrapper(env) # needs to be last

    return env