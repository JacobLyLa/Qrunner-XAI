
import random
import time
from collections import deque

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pygame
from moviepy.editor import ImageSequenceClip
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from src.Qrunner.qrunner import QrunnerEnv
from src.Qrunner.qrunner_modified import ModifiedQrunnerEnv

# Most of the wrappers are based on:
# (https://gymnasium.farama.org/api/wrappers/)
# (https://stable-baselines3.readthedocs.io/en/master/common/atari_wrappers.html)

class LimitEnv(gym.Wrapper):
    def __init__(self,env, max_steps, max_steps_no_reward):
        super().__init__(env)
        self._max_steps = max_steps
        self._max_steps_no_reward = max_steps_no_reward
        self._elapsed_steps = 0
        self._elapsed_steps_no_reward = 0

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1
        
        if reward == 0:
            self._elapsed_steps_no_reward += 1
        else:
            self._elapsed_steps_no_reward = 0

        if self._elapsed_steps >= self._max_steps or self._elapsed_steps_no_reward >= self._max_steps_no_reward:
            truncated = True

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        self._elapsed_steps_no_reward = 0
        return self.env.reset(**kwargs)
    
class FrameBlending(gym.Wrapper):
    def __init__(self, env, alpha=0.8):
        super().__init__(env)
        self.env = env
        self.alpha = alpha
        self.previous_obs = None

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        # Blend if previous observation exists, otherwise just use the current observation
        if self.previous_obs is not None:
            new_observation = self.alpha * observation + (1 - self.alpha) * self.previous_obs
            new_observation = np.clip(new_observation, 0, 255).astype(np.uint8)
        else:
            new_observation = observation
        self.previous_obs = observation.copy()
        return new_observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.previous_obs = observation.copy()
        return observation, info

class SkipEnv(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        last_obs = None
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            last_obs = obs
            total_reward += float(reward)
            if terminated or truncated:
                break

        return last_obs, total_reward, terminated, truncated, info

class HumanRendering(gym.Wrapper):
    def __init__(self, env, scale=1, record_video=False):
        super().__init__(env)
        self.scale = scale
        self.record_video = record_video
        self.screen_size = None
        self.window = None
        self.clock = None
        self.frames = []

    def step(self, *args, **kwargs):
        result = self.env.step(*args, **kwargs)
        return result

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        self._render_frame()
        if len(self.frames) > 1 and self.record_video:
            clip = ImageSequenceClip(self.frames, fps=60)
            clip.write_videofile(f"videos/episode_{time.time()}.mp4")
            self.frames = []
        return result

    def render(self):
        return None

    def _render_frame(self):
        rgb_array = self.env.render()
        rgb_array = np.transpose(rgb_array, axes=(1, 0, 2))

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
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()
        
        array = pygame.surfarray.array3d(surf)
        array = np.transpose(array, (1, 0, 2))
        self.frames.append(array)

    def close(self):
        super().close()
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

class RenderWrapper(gym.Wrapper):
    def __init__(self, env, length, render_salient=False, plot_q=False):
        gym.Wrapper.__init__(self, env)
        self.render_salient = render_salient
        self.plot_q = plot_q
        self.unwrapped.q_values = deque(maxlen=length)
        self.unwrapped.push_q_values = self.push_q_values
        self.max_scale = 1
        self.min_scale = 0
        self.unwrapped.gradient = None
        self.unwrapped.set_gradient = self.set_gradient
        
        self.initialized = False

    def set_gradient(self, gradient):
        self.unwrapped.gradient = gradient
        
    def push_q_values(self, q_values):
        if not self.plot_q:
            return
        self.unwrapped.q_values.append(q_values)
        self.max_scale = max(max(q_values)*1.5, self.max_scale)
        # self.min_scale = min(min(q_values), self.min_scale)
        if not self.initialized:
            self.initialize_plot()
        self.update_plot()
        
    def initialize_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        
        self.max_q_value_line, = self.ax.plot([], [], label='max Q-value', linewidth=2, color='blue')
        
        self.ax.legend(loc='lower right')
        self.ax.set_title("Max Q-Value Over Time")
        self.ax.set_xlabel("Time Step")
        self.ax.set_ylabel("Max Q-Value")
        self.initialized = True
        
    def update_plot(self):
        # Set y limits
        self.ax.set_ylim([self.min_scale, self.max_scale])
        
        time_steps = range(len(self.unwrapped.q_values))
        all_q_values = np.array(self.unwrapped.q_values)

        # Calculate max Q-value across all actions for each time step
        max_q_values = np.max(all_q_values, axis=1)
        self.max_q_value_line.set_xdata(time_steps)
        self.max_q_value_line.set_ydata(max_q_values)

        # Adjust plot limits and redraw
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def render(self):
        obs = self.env.render()
        if self.unwrapped.gradient is not None and self.render_salient:
            obs = self.overlay_salience_map(obs)
        return obs
    
    def overlay_salience_map(self, obs):
        if self.unwrapped.gradient is None:
            return obs
        
        gradient = self.unwrapped.gradient
        positive_gradient = gradient.clip(min=0).sum(axis=2)
        negative_gradient = np.abs(gradient.clip(max=0).sum(axis=2))
        
        height, width, _ = obs.shape
        saliency_height, saliency_width = positive_gradient.shape
        # Rescale if necessary
        if (height, width) != (saliency_height, saliency_width):
            positive_gradient = cv2.resize(positive_gradient, (width, height))
            negative_gradient = cv2.resize(negative_gradient, (width, height))
        
        positive_gradient = gaussian_filter(positive_gradient, sigma=6)
        positive_gradient = positive_gradient * 20000
        positive_gradient = positive_gradient.clip(0, 1)
        positive_gradient[positive_gradient < 0.3] = 0
        
        negative_gradient = gaussian_filter(negative_gradient, sigma=6)
        negative_gradient = negative_gradient * 20000
        negative_gradient = negative_gradient.clip(0, 1)
        negative_gradient[negative_gradient < 0.3] = 0

        # Create a 3-channel color map for positive gradient in green
        pos_map = np.zeros_like(obs)
        pos_map[:, :, 2] = positive_gradient * 255  # Green channel

        # Create a 3-channel color map for negative gradient in red
        neg_map = np.zeros_like(obs)
        neg_map[:, :, 0] = negative_gradient * 255  # Red channel

        # Overlay the positive and negative maps on the original image with transparency
        alpha = 0.4  # Transparency factor
        obs_with_pos = cv2.addWeighted(obs, 1, pos_map.astype(np.uint8), alpha, 0)
        final_obs = cv2.addWeighted(obs_with_pos, 1, neg_map.astype(np.uint8), alpha, 0)

        return final_obs

def wrapped_qrunner_env(frame_skip, max_steps=5000, max_steps_no_reward=50, human_render=False, render_salient=False, plot_q=False, record_video=False, scale=8):
    # If salient or plot q is true, then assert that human render is also true
    assert not (render_salient or plot_q) or human_render
    original = True
    if original:
        env = QrunnerEnv()
    else:
        env = ModifiedQrunnerEnv()
    if human_render:
        env = RenderWrapper(env, length=100, plot_q=plot_q, render_salient=render_salient)
        env = HumanRendering(env, scale=scale, record_video=record_video)
    
    if frame_skip > 1:
        env = SkipEnv(env, skip=frame_skip)
    
    env = LimitEnv(env, max_steps=max_steps, max_steps_no_reward=max_steps_no_reward)
    env = FrameBlending(env)
    
    env = gym.wrappers.AutoResetWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

def main():
    env = wrapped_qrunner_env(frame_skip=3, human_render=False, record_video=False, scale=6)
    obs, _ = env.reset()
    num_frames = 50
    save_path = 'figures/observations/'

    for i in tqdm(range(num_frames), desc="Processing frames"):
        action = 3 if random.random() < 0.2 else 2 # move right and sometimes jump
        obs, rewards, terminated, truncated, info = env.step(action)
        # Save the observation as an image
        if True:
            plt.imshow(obs.astype(np.uint8))
            plt.title(f"Frame {i}")
            cbar = plt.colorbar(orientation='horizontal')
            plt.tight_layout()
            plt.savefig(f"{save_path}frame_{i}.png")
            plt.close()
    env.close()

    print(f"Frames saved to: {save_path}")

if __name__ == "__main__":
    main()