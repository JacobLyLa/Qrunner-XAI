
import random
import time
from collections import deque

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pygame
from gymnasium.spaces import Box
from gymnasium.wrappers import TransformReward
from moviepy.editor import ImageSequenceClip
from scipy.ndimage import gaussian_filter
from stable_baselines3.common.env_checker import check_env
from tqdm import tqdm

from src.Qrunner.qrunner import QrunnerEnv

# Most of the wrappers are based on:
# (https://gymnasium.farama.org/api/wrappers/)
# (https://stable-baselines3.readthedocs.io/en/master/common/atari_wrappers.html)

class LazyFrames:
    def __init__(self, frames, lz4_compress=False):
        self.frame_shape = frames[0].shape
        self.shape = (self.frame_shape[0], self.frame_shape[1], self.frame_shape[2] * len(frames))
        self.dtype = frames[0].dtype
        if lz4_compress:
            try:
                from lz4.block import compress
                frames = [compress(frame) for frame in frames]
            except ImportError:
                raise gym.error.DependencyNotInstalled("lz4 is not installed, run `pip install gym[other]`")
        self._frames = frames
        self.lz4_compress = lz4_compress

    def __array__(self, dtype=None):
        frames = np.concatenate([self._check_decompress(f) for f in self._frames], axis=-1)
        if dtype is not None:
            frames = frames.astype(dtype)
        return frames

    def _check_decompress(self, frame):
        if self.lz4_compress:
            from lz4.block import decompress
            return np.frombuffer(decompress(frame), dtype=self.dtype).reshape(self.frame_shape)
        return frame

class FrameStack(gym.ObservationWrapper):
    def __init__(self, env, num_stack, lz4_compress=False):
        super().__init__(env)
        self.num_stack = num_stack
        self.lz4_compress = lz4_compress
        self.frames = deque(maxlen=num_stack)

        # Original frame shape
        frame_shape = env.observation_space.shape
        channel_count = frame_shape[2]

        # New shape for the stacked frames
        new_shape = (frame_shape[0], frame_shape[1], channel_count * num_stack)
        self.observation_space = Box(
            low=np.min(env.observation_space.low) * np.ones(new_shape, dtype=env.observation_space.dtype),
            high=np.max(env.observation_space.high) * np.ones(new_shape, dtype=env.observation_space.dtype),
            dtype=env.observation_space.dtype,
        )

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(observation)
        return self.observation(None), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self.observation(None), info

    def observation(self, observation):
        return LazyFrames(list(self.frames), self.lz4_compress)

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
    def __init__(self, env, scale=1):
        super().__init__(env)
        self.scale = scale
        self.screen_size = None
        self.window = None
        self.clock = None
        self.frames = []
        
    def step(self, *args, **kwargs):
        result = self.env.step(*args, **kwargs)
        self._render_frame()
        return result

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        self._render_frame()
        if len(self.frames) > 1:
            clip = ImageSequenceClip(self.frames, fps=60)
            clip.write_videofile(f"videos/episode_{time.time()}.mp4")
            clip.speedx(4).write_gif(f"videos/episode_{time.time()}.gif")
            self.frames = []
        return result

    def render(self):
        return None

    def _render_frame(self):
        rgb_array = self.env.render()
        rgb_array = np.transpose(rgb_array, axes=(1, 0, 2))

        if self.screen_size is None:
            self.screen_size = (rgb_array.shape[0] * self.scale, rgb_array.shape[1] * self.scale)
            print(f"Screen size: {self.screen_size}")

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.screen_size)

        if self.clock is None:
            self.clock = pygame.time.Clock()
            print(f"Clock: {self.clock}")

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
            import pygame

            pygame.display.quit()
            pygame.quit()

class RenderWrapper(gym.Wrapper):
    def __init__(self, env, length):
        gym.Wrapper.__init__(self, env)
        self.unwrapped.q_values = deque(maxlen=length)
        self.unwrapped.push_q_values = self.push_q_values
        self.max_scale = 1
        self.min_scale = 0
        self.unwrapped.salience_map = None
        self.unwrapped.set_salience = self.set_salience
        
        self.initialized = False
        print(env.get_action_meanings())

    def set_salience(self, saliency_map):
        self.unwrapped.salience_map = saliency_map
        
    def push_q_values(self, q_values):
        self.unwrapped.q_values.append(q_values)
        self.max_scale = max(max(q_values), self.max_scale)
        self.min_scale = min(min(q_values), self.min_scale)
        if not self.initialized:
            self.initialize_plot()
        self.update_plot()
        
    def initialize_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.q_value_lines = []
        action_meanings = self.env.unwrapped.get_action_meanings()
        for action in action_meanings:
            line, = self.ax.plot([], [], label=action)
            self.q_value_lines.append(line)
        # Lines for mean and max Q-values
        # self.mean_q_value_line, = self.ax.plot([], [], label='Mean Q-value')
        self.max_q_value_line, = self.ax.plot([], [], label='Max Q-value')
        self.ax.legend()
        self.ax.set_title("Q Values Over Time")
        self.ax.set_xlabel("Time Step")
        self.ax.set_ylabel("Q Value")
        self.initialized = True
        
    def update_plot(self):
        # Set y limits
        self.ax.set_ylim([self.min_scale, self.max_scale])
        
        time_steps = range(len(self.unwrapped.q_values))
        all_q_values = np.array(self.unwrapped.q_values)

        '''
        # Update line for mean Q-value
        mean_q_values = np.mean(all_q_values, axis=1)
        self.mean_q_value_line.set_xdata(time_steps)
        self.mean_q_value_line.set_ydata(mean_q_values)
        self.mean_q_value_line.set_linewidth(3)  # Make the line a bit bolder
        '''

        # Update line for max Q-value
        max_q_values = np.max(all_q_values, axis=1)
        self.max_q_value_line.set_xdata(time_steps)
        self.max_q_value_line.set_ydata(max_q_values)
        self.max_q_value_line.set_linewidth(5)  # Make the line a bit bolder

        # Adjust plot limits and redraw
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.legend(loc='upper left')  # Fix the legend position
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Update lines for each action
        for i, line in enumerate(self.q_value_lines):
            line.set_xdata(time_steps)
            line.set_ydata(all_q_values[:, i])

    def render(self):
        obs = self.env.render()
        if self.unwrapped.salience_map is not None:
            obs = self.overlay_salience_map(obs)
        return obs
    
    # TODO: Choose between
    # 1. Red bad and green good, look at gradient sign
    # 2. Look at gradient magnitude, blue -> red
    def overlay_salience_map(self, obs):
        if self.unwrapped.salience_map is None:
            return obs
        
        gradients = self.unwrapped.salience_map
        positive_gradients = gradients.clip(min=0).sum(axis=2)
        negative_gradients = np.abs(gradients.clip(max=0).sum(axis=2))
        
        height, width, _ = obs.shape
        saliency_height, saliency_width = positive_gradients.shape
        # Rescale if necessary
        if (height, width) != (saliency_height, saliency_width):
            positive_gradients = cv2.resize(positive_gradients, (width, height))
            negative_gradients = cv2.resize(negative_gradients, (width, height))
        
        positive_gradients = gaussian_filter(positive_gradients, sigma=6)
        positive_gradients = positive_gradients * 20000
        positive_gradients = positive_gradients.clip(0, 1)
        positive_gradients[positive_gradients < 0.3] = 0
        
        negative_gradients = gaussian_filter(negative_gradients, sigma=6)
        negative_gradients = negative_gradients * 20000
        negative_gradients = negative_gradients.clip(0, 1)
        negative_gradients[negative_gradients < 0.3] = 0

        # Create a 3-channel color map for positive gradients in green
        pos_map = np.zeros_like(obs)
        pos_map[:, :, 2] = positive_gradients * 255  # Green channel

        # Create a 3-channel color map for negative gradients in red
        neg_map = np.zeros_like(obs)
        neg_map[:, :, 0] = negative_gradients * 255  # Red channel

        # Overlay the positive and negative maps on the original image with transparency
        alpha = 0.4  # Transparency factor
        obs_with_pos = cv2.addWeighted(obs, 1, pos_map.astype(np.uint8), alpha, 0)
        final_obs = cv2.addWeighted(obs_with_pos, 1, neg_map.astype(np.uint8), alpha, 0)

        return final_obs
    '''
    def overlay_salience_map(self, obs):
        if self.unwrapped.salience_map is None:
            return obs
        
        gradients = self.unwrapped.salience_map
        
        height, width, _ = obs.shape
        saliency_height, saliency_width = gradients.shape
        # Rescale if necessary
        if (height, width) != (saliency_height, saliency_width):
            salience_map = cv2.resize(gradients, (width, height))
        else:
            salience_map = gradients
        
        salience_map = gaussian_filter(salience_map, sigma=6)
        # salience_map = salience_map / np.max(salience_map)
        salience_map = salience_map * 20000
        salience_map = salience_map.clip(0, 1)
        salience_map[salience_map < 0.6] = 0

        # Convert obs from RGB to BGR
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)

        # Apply the colormap to the salience map
        salience_map = cv2.applyColorMap(np.uint8(salience_map * 255), cv2.COLORMAP_JET)
        obs = cv2.addWeighted(obs, 1, salience_map, 0.2, 0)

        # back to RGB afterwards
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)

        return obs
    '''


def wrapped_qrunner_env(frame_skip=3, frame_stack=2, max_steps=5000, max_steps_no_reward=50, human_render=False, scale=8):
    env = QrunnerEnv()
    if human_render:
        env = RenderWrapper(env, length=100)
        
    if human_render:
        env = HumanRendering(env, scale=scale)
    
    if frame_skip > 1:
        env = SkipEnv(env, skip=frame_skip)
    
    env = LimitEnv(env, max_steps=max_steps, max_steps_no_reward=max_steps_no_reward)
    
    if frame_stack > 1:
        env = FrameStack(env, frame_stack)
    elif frame_stack == 1:
        env = FrameBlending(env)
    
    env = gym.wrappers.AutoResetWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # TODO: need this?
    #env = TransformReward(env, lambda r: r-0.0001)
    return env

def convert_obs_to_image(obs):
    # obs is (84, 84, 12)
    single_frame_height, single_frame_width, channels = obs.shape
    num_frames = channels // 3

    # New shape (4, 84, 84, 3)
    reshaped_obs = obs.reshape(single_frame_height, single_frame_width, num_frames, 3).transpose(2, 0, 1, 3)

    # New shape (84 * 4, 84 * 3, 3)
    image = np.zeros((single_frame_height * num_frames, single_frame_width * 3, 3))
    for i in range(num_frames):
        for j in range(3):
            image[i * single_frame_height:(i + 1) * single_frame_height, j * single_frame_width:(j + 1) * single_frame_width, :] = reshaped_obs[i, :, :, j][:, :, np.newaxis]

    return image

def main():
    env = wrapped_qrunner_env(frame_skip=3, frame_stack=1, human_render=False, scale=6)
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