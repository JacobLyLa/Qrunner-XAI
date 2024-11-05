import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pygame
from tqdm import tqdm
from src.Qrunner.qrunner import QrunnerEnv
from src.Qrunner.qrunner_modified import ModifiedQrunnerEnv
import cv2  # Make sure to install OpenCV: pip install opencv-python
from gymnasium import spaces

class QrunnerWrapper(gym.Wrapper):
    def __init__(self, env, max_steps, max_steps_reward, blending_alpha, frame_skip, use_grayscale=False):
        super().__init__(env)
        self.max_steps = max_steps
        self.max_steps_reward = max_steps_reward
        self.blending_alpha = blending_alpha
        self.frame_skip = frame_skip
        self.use_grayscale = use_grayscale  # New attribute
        channels = 1 if use_grayscale else 3
        self.observation_space = spaces.Box(low=0, high=255, shape=(channels, 84, 84), dtype=np.uint8)

        self.previous_obs = None
        self.total_reward = 0
        self.steps = 0
        self.steps_without_reward = 0

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        # Handle blending
        observation = self.blend_frames(observation)
        self.previous_obs = observation.copy() if not self.use_grayscale else observation  # Ensure correct shape

        # Convert to grayscale if enabled
        if self.use_grayscale:
            observation = self.to_grayscale(observation)

        # Handle frame skipping
        sum_reward = reward
        for _ in range(self.frame_skip - 1):
            if terminated or truncated:
                break
            observation, reward, terminated, truncated, info = self.env.step(action)
            sum_reward += reward
            if self.use_grayscale:
                observation = self.to_grayscale(observation)
        self.total_reward += sum_reward

        # Handle max steps
        self.steps += 1
        if sum_reward == 0:
            self.steps_without_reward += 1
        else:
            self.steps_without_reward = 0
        if self.steps >= self.max_steps or self.steps_without_reward >= self.max_steps_reward:
            truncated = True

        if terminated or truncated:
            info["total_reward"] = self.total_reward
            info["wrapper_steps"] = self.steps
        return observation, sum_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.steps = 0
        self.steps_without_reward = 0
        self.previous_obs = None
        self.total_reward = 0
        observation, info = self.env.reset(**kwargs)
        if self.use_grayscale:
            observation = self.to_grayscale(observation)
        return observation, info

    def blend_frames(self, observation):
        if self.blending_alpha == 1.0 or self.previous_obs is None:
            return observation
        blended = np.clip(
            self.blending_alpha * observation + (1 - self.blending_alpha) * self.previous_obs, 
            0, 
            255
        ).astype(np.uint8)
        return blended

    def to_grayscale(self, observation):
        grayscale = cv2.cvtColor(observation.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
        # Add channel dimension
        return np.expand_dims(grayscale, axis=0)  # Shape: (1, H, W)

class HumanRenderWrapper(gym.Wrapper):
    def __init__(self, env, scale=6, fps=10, wrapped_render=False):
        super().__init__(env)
        self.scale = scale
        self.screen_size = None
        self.window = None
        self.clock = None
        self.fps = fps
        self.wrapped_render = wrapped_render

    def step(self, *args, **kwargs):
        result = self.env.step(*args, **kwargs)
        self._render_frame()
        return result

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        return result

    def close(self):
        super().close()
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _render_frame(self):
        if self.wrapped_render: # Still scaled and renormalized
            rgb_array = np.clip(self.previous_obs * 255.0, 0, 255).astype(np.uint8)
        else:
            rgb_array = self.env.render()
        rgb_array = np.transpose(rgb_array, axes=(1, 2, 0))

        if self.screen_size is None:
            self.screen_size = (rgb_array.shape[0] * self.scale, rgb_array.shape[1] * self.scale)

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.screen_size)

        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = pygame.surfarray.make_surface(rgb_array)
        if self.scale != 1:
            surf = pygame.transform.scale(surf, self.screen_size)
        self.window.blit(surf, (0, 0))
        pygame.event.pump()
        self.clock.tick(self.fps)
        pygame.display.flip()

def main():
    import time
    
    env = QrunnerEnv()
    env = QrunnerWrapper(env, max_steps=100, max_steps_reward=500, blending_alpha=0.5, frame_skip=3, use_grayscale=True)
    #env = HumanRenderWrapper(env, scale=6, wrapped_render=True)
    obs, _ = env.reset()

    for i in range(500):
        action = 3 if random.random() < 0.2 else 2 # move right and sometimes jump
        obs, rewards, terminated, truncated, info = env.step(action)
        print(f"Action: {action} | Obs: {obs.shape} | Rewards: {rewards} | Terminated: {terminated} | Truncated: {truncated} | Info: {info}")
        time.sleep(0.1)
        if terminated or truncated:
            break
    env.close()

if __name__ == "__main__":
    main()