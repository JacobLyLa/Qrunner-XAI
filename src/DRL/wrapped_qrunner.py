
import random
import time
from collections import deque

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pygame
from gymnasium.wrappers import (ResizeObservation, TimeLimit,
                                TransformReward)
from stable_baselines3.common.atari_wrappers import (ClipRewardEnv,
                                                     NoopResetEnv, WarpFrame)
from stable_baselines3.common.env_checker import check_env
from tqdm import tqdm
from gymnasium.spaces import Box
from src.Qrunner.qrunner import QrunnerEnv

# Most of the wrappers are based on:
# gymnasium.wrappers (src)
# stable_baselines3.common.atari_wrappers (src)

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

    def step(self, *args, **kwargs):
        result = self.env.step(*args, **kwargs)
        self._render_frame()
        return result

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        self._render_frame()
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

    def close(self):
        super().close()
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()


# TODO: New window with graphs
# But this method can still be used for saliency maps?
class RenderWrapper(gym.Wrapper):
    def __init__(self, env, length):
        gym.Wrapper.__init__(self, env)
        self.unwrapped.q_values = deque(maxlen=length)
        self.unwrapped.push_q_values = self.push_q_values
        self.max_scale = 1
        self.min_scale = 0
        self.unwrapped.salience_map = None
        self.unwrapped.set_salience = self.set_salience

    def set_salience(self, saliency_map):
        self.unwrapped.salience_map = saliency_map
        
    def push_q_values(self, q_values):
        self.unwrapped.q_values.append(q_values)
        self.max_scale = max(max(q_values), self.max_scale)
        self.min_scale = min(min(q_values), self.min_scale)

    def render(self):
        obs = self.env.render()

        if len(self.unwrapped.q_values) > 0:
            obs = self.overlay_q_values_graph(obs)
            
        if self.unwrapped.salience_map is not None:
            obs = self.overlay_salience_map(obs)
        return obs

    def overlay_q_values_graph(self, obs):
        # Get the dimensions of the observation
        height, width, _ = obs.shape
        num_actions = len(self.unwrapped.q_values[0])
        num_steps = len(self.unwrapped.q_values)
        y_center = int(height * 0.1)
        y_center_deviation = int(height * 0.1)
        x_end = int(width * 0.5)
        # For each step
        for i in range(1, num_steps):
            mean = 0
            mean_prev = 0
            # For each action
            for j in range(num_actions):
                q_value_scaled = self.scale_q_value(self.unwrapped.q_values[i][j])
                q_value_scaled_prev = self.scale_q_value(self.unwrapped.q_values[i - 1][j])
                mean += q_value_scaled
                mean_prev += q_value_scaled_prev
                '''
                x1 = int(x_end * (i - 1) / num_steps)
                y1 = int(y_center - q_value_scaled_prev * y_center_deviation)
                x2 = int(x_end * i / num_steps)
                y2 = int(y_center - q_value_scaled * y_center_deviation)
            
                self.draw_line(obs, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=1)
                '''
            mean /= num_actions
            mean_prev /= num_actions
            x1 = int(x_end * (i - 1) / num_steps)
            y1 = int(y_center - mean_prev * y_center_deviation)
            x2 = int(x_end * i / num_steps)
            y2 = int(y_center - mean * y_center_deviation)
            self.draw_line(obs, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)

        # Draw boundary box
        self.draw_line(obs, (0, y_center + y_center_deviation), (x_end, y_center + y_center_deviation), color=(255, 255, 255))
        self.draw_line(obs, (0, y_center - y_center_deviation), (x_end, y_center - y_center_deviation), color=(255, 255, 255))
        self.draw_line(obs, (x_end, y_center - y_center_deviation), (x_end, y_center + y_center_deviation), color=(255, 255, 255))
        # Draw positive/negative seperation line
        zero_scaled = self.scale_q_value(0)
        y_zero = int(y_center - zero_scaled * y_center_deviation)
        self.draw_line(obs, (0, y_zero), (x_end, y_zero), color=(0, 0, 255), thickness=1)

        return obs
    
    # TODO: Probably tons of ways to visualize this.
    # 1. Red bad and green good, look at gradient sign
    # 2. Look at gradient magnitude, blue -> red 
    def overlay_salience_map(self, obs):
        if self.unwrapped.salience_map is None:
            return obs

        height, width, _ = obs.shape
        saliency_height, saliency_width = self.unwrapped.salience_map.shape

        # Rescale if necessary
        if (height, width) != (saliency_height, saliency_width):
            salience_map = cv2.resize(self.unwrapped.salience_map, (width, height))
        else:
            salience_map = self.unwrapped.salience_map
        
        salience_map = salience_map / np.max(salience_map)

        # Convert obs from RGB to BGR
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)

        # Apply the colormap to the salience map
        salience_map = cv2.applyColorMap(np.uint8(salience_map * 255), cv2.COLORMAP_JET)
        obs = cv2.addWeighted(obs, 1, salience_map, 0.2, 0)

        # back to RGB afterwards
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)

        return obs

    # Scale to [-1, 1]
    def scale_q_value(self, q):
        return (q - self.min_scale) * 2 / (self.max_scale - self.min_scale) - 1

    def draw_line(self, img, start, end, color=(255, 255, 255), thickness=1):
        # Bresenham's line algorithm to draw a line on the image
        x1, y1 = start
        x2, y2 = end
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            img[y1:y1+thickness, x1:x1+thickness] = color
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
                
    def reset(self, seed=None, options=None):
        self.unwrapped.q_values.clear()
        return self.env.reset(seed=seed, options=options)
    

def wrapped_qrunner_env(frame_skip=3, frame_stack=2, max_steps=5000, max_steps_no_reward=50, human_render=False, record_video=False, scale=8):
    env = QrunnerEnv()
    if human_render or record_video:
        env = RenderWrapper(env, length=100)
        
    if human_render:
        env = HumanRendering(env, scale=scale)
        
    # TODO: Align with checkpoints from training and use scale
    if record_video:
        checkpoints = [0, 10, 100, 1000, 10000, 100000, 1000000]
        env = gym.wrappers.RecordVideo(env=env, name_prefix='recording', video_folder=f"./videos", episode_trigger=lambda x: x in checkpoints)
    
    if frame_skip > 1:
        # Skips frames and repeats action and sums reward. max not needed
        env = SkipEnv(env, skip=frame_skip)
    
    LimitEnv(env, max_steps=max_steps, max_steps_no_reward=max_steps_no_reward)
    
    env = FrameStack(env, frame_stack)
    
    env = gym.wrappers.AutoResetWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = TransformReward(env, lambda r: r-0.0001)
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
    env = wrapped_qrunner_env(frame_skip=3, frame_stack=4, human_render=True, record_video=False, scale=6)
    #check_env(env, warn=True)
    
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation space: {env.observation_space}")
    num_frames = 50
    save_path = 'figures/observations/'

    for i in tqdm(range(num_frames), desc="Processing frames"):
        action = 3 if random.random() < 0.2 else 2 # move right and sometimes jump
        obs, rewards, terminated, truncated, info = env.step(action)
        # Save the observation as an image
        if True:
            obs = np.array(obs)
            obs = convert_obs_to_image(obs).astype(np.uint8)
            plt.imshow(obs)
            plt.title(f"Frame {i}")
            cbar = plt.colorbar(orientation='horizontal')
            plt.tight_layout()
            plt.savefig(f"{save_path}frame_{i}.png")
            plt.close()
    env.close()

    print(f"Frames saved to: {save_path}")

if __name__ == "__main__":
    main()