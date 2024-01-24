
import random
import time
from collections import deque

import matplotlib

matplotlib.use('Agg')
import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.wrappers import (FrameStack, HumanRendering, ResizeObservation,
                                TimeLimit, TransformReward)
from stable_baselines3.common.atari_wrappers import (ClipRewardEnv,
                                                     MaxAndSkipEnv,
                                                     NoopResetEnv, WarpFrame)
from stable_baselines3.common.env_checker import check_env
from tqdm import tqdm

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

# TODO: let's just open a new window with the graph, much cleaner
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
        # Update dynamic scale
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
        y_center = int(height * 0.8)
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
                x1 = int(x_end * (i - 1) / num_steps)
                y1 = int(y_center - q_value_scaled_prev * y_center_deviation)
                x2 = int(x_end * i / num_steps)
                y2 = int(y_center - q_value_scaled * y_center_deviation)
            
                self.draw_line(obs, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=1)
            mean /= num_actions
            mean_prev /= num_actions
            x1 = int(x_end * (i - 1) / num_steps)
            y1 = int(y_center - mean_prev * y_center_deviation)
            x2 = int(x_end * i / num_steps)
            y2 = int(y_center - mean * y_center_deviation)
            self.draw_line(obs, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=3)

        # Draw boundary box
        self.draw_line(obs, (0, y_center + y_center_deviation), (x_end, y_center + y_center_deviation), color=(255, 255, 255))
        self.draw_line(obs, (0, y_center - y_center_deviation), (x_end, y_center - y_center_deviation), color=(255, 255, 255))
        self.draw_line(obs, (x_end, y_center - y_center_deviation), (x_end, y_center + y_center_deviation), color=(255, 255, 255))
        # Draw positive/negative seperation line
        zero_scaled = self.scale_q_value(0)
        y_zero = int(y_center - zero_scaled * y_center_deviation)
        self.draw_line(obs, (0, y_zero), (x_end, y_zero), color=(0, 0, 255), thickness=2)

        return obs
    
    # TODO: Probably tons of ways to visualize this.
    # 1. Red bad and green good, look at gradient sign
    # 2. Look at gradient magnitude, blue -> red 
    def overlay_salience_map(self, obs):
        if self.unwrapped.salience_map is None:
            return obs

        height, width, _ = obs.shape
        saliency_height, saliency_width = self.unwrapped.salience_map.shape

        # Rescale the salience map to match the observation size if necessary
        if (height, width) != (saliency_height, saliency_width):
            salience_map_resized = cv2.resize(self.unwrapped.salience_map, (width, height))
        else:
            salience_map_resized = self.unwrapped.salience_map

        '''
        # Normalize the salience values to the range [0, 1]
        salience_map_normalized = cv2.normalize(salience_map_resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        # Calculate percentiles
        percentile_high = np.percentile(salience_map_normalized, 95)
        percentile_low = np.percentile(salience_map_normalized, 5)

        # Create masks based on the calculated percentiles
        mask_above_threshold = salience_map_normalized >= percentile_high
        mask_below_threshold = salience_map_normalized <= percentile_low

        # Apply the threshold to determine where to overlay colors
        obs[:, :, 0] = np.where(mask_below_threshold, np.clip(obs[:, :, 0] + (1 - salience_map_normalized) * 255, 0, 255), obs[:, :, 0])
        obs[:, :, 1] = np.where(mask_above_threshold, np.clip(obs[:, :, 1] + (salience_map_normalized) * 255, 0, 255), obs[:, :, 1])
        '''
        
        # Shift values so more extreme values are more visible
        #salience_map_resized = np.power(salience_map_resized, 2.0)
        # Transform to absolute and divide by the maximum
        salience_map_absolute = np.abs(salience_map_resized)
        salience_map_scaled = salience_map_absolute / np.max(salience_map_absolute)

        # Convert obs from RGB to BGR
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)

        # Apply the colormap to the salience map
        salience_map_colored = cv2.applyColorMap(np.uint8(salience_map_scaled * 255), cv2.COLORMAP_JET)
        obs = cv2.addWeighted(obs, 1, salience_map_colored, 0.4, 0)

        # ack to RGB afterwards
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
    

# TODO: one giant custom wrapper with everything combined (maybe not framestack though)
def wrapped_qrunner_env(size, frame_skip, frame_stack, record_video=False, human_render=False):
    assert not (record_video and human_render), "Can not use record video and human render together"
    
    # Size is actual env size (for human render or video), while target_size is the rescaled size (for agent)
    # TODO: more efficient to create env in target size, then upscale for human render and video
    target_size = 84
    env = QrunnerEnv(size=size)
    
    # TODO: 
    # Create own so own text, graphs, and even resized observations can be plotted.
    # That would probably allow for video recording at the same time too.
    # https://github.com/openai/gym/blob/master/gym/wrappers/human_rendering.py
    if human_render or record_video:
        env = RenderWrapper(env, length=100)
        
    if human_render:
        env = HumanRendering(env)

    if record_video:
        # TODO: align with checkpoints from training
        checkpoints = [0, 10, 100, 1000, 10000, 100000, 1000000]
        env = gym.wrappers.RecordVideo(env=env, name_prefix='recording', video_folder=f"./videos", episode_trigger=lambda x: x in checkpoints)
    
    if frame_skip > 1:
        # Skips frames and repeats action and sums reward. max not needed
        env = MaxAndSkipEnv(env, skip=frame_skip)  
    
    # This is after max and skip, so 500 frames for the agent, not necessarily 500 frames for the env
    TimeLimit(env, max_episode_steps=500) 
    
    #env = WarpFrame(env, width=size, height=size)  # Resize and grayscale
    
    if size != target_size:
        env = ResizeObservation(env, shape=(target_size, target_size))
    
    # Stack frames, SB3 doesn't allow cnn policy with obs: (x, 42, 42, 3)
    env = FrameStack(env, frame_stack)
    # TODO: test. but then walking right and collecting coin etc is equal?
    #env = ClipRewardEnv(env)
    
    env = Reshape(env, target_size, frame_stack, 3)

    env = gym.wrappers.AutoResetWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    # env = TransformReward(env, lambda r: r-0.0000001)

    return env

def main():
    env = wrapped_qrunner_env(size=84*6, frame_skip=4, frame_stack=4, record_video=False, human_render=True)
    #check_env(env, warn=True)
    
    obs = env.reset()
    num_frames = 100
    save_path = 'figures/observations/'

    for i in tqdm(range(num_frames), desc="Processing frames"):
        action = 3 if random.random() < 0.2 else 2 # move right and sometimes jump
        obs, rewards, terminated, truncated, info = env.step(action)
        # Save the observation as an image
        if True:
            obs = np.concatenate(obs, axis=1)
            plt.imshow(obs, cmap='gray')
            plt.title(f"Frame {i}")
            cbar = plt.colorbar(orientation='horizontal')
            plt.savefig(f"{save_path}frame_{i}.png")
            plt.close()
    env.close()

    print(f"Frames saved to: {save_path}")

if __name__ == "__main__":
    main()