import pickle
import random

import gymnasium as gym
import numpy as np

from utils import prepare_folders


class GameStep:
    def __init__(self, observation, image, state_variables):
        self.observation = np.array(observation).copy()
        if image is not None: # when render mode is human, image is None
            self.image = image.copy()
        self.state_variables = state_variables.copy()

# rename to state recorder?
class GameStepSaverWrapper(gym.Wrapper):
    # RAM map used from
    # https://github.com/mila-iqia/atari-representation-learning/blob/master/atariari/benchmark/ram_annotations.py
    ram_map = {
        'ball_x': 99,
        'ball_y': 101,
        'paddle_x': 72,
    }
    def __init__(self, env, save_interval):
        gym.Wrapper.__init__(self, env)
        self.env.state_variables = None
        self.game_steps = []
        self.save_interval = save_interval
        self.step_counter = 0
        self.episode_frame_number = 0

        self.state_variables = {
            'ball_x': 0,
            'ball_y': 0,
            'ball_vx': 0,
            'ball_vy': 0,
            'paddle_x': 0,
            'lives': 1,
            'losing life': False,
            'bricks_hit': 0,
            'collision': False,
            'brick_hit': False,
        }

    def get_ram_value(self, variable_name):
        ram = self.env.unwrapped.ale.getRAM()
        return int(ram[GameStepSaverWrapper.ram_map[variable_name]])

    def step(self, action):
        self.observation, self.reward, self.termination, self.truncation, self.info = self.env.step(action)
        # read state variables from RAM and env
        ball_x = self.get_ram_value('ball_x') - 49
        ball_y = self.get_ram_value('ball_y') + 10
        paddle_x = self.get_ram_value('paddle_x') - 39
        lives = self.env.unwrapped.ale.lives()

        if ball_y >= 194:
            losing_life = True
        else:
            losing_life = False

        # check if new episode
        if self.info['episode_frame_number'] < self.episode_frame_number:
            self.state_variables['bricks_hit'] = 0
        self.episode_frame_number = self.info['episode_frame_number']

        ball_vx = ball_x - self.state_variables['ball_x']
        ball_vy = ball_y - self.state_variables['ball_y']

        # if velocity changed, then there was a collision
        collision = False
        if ball_vx != self.state_variables['ball_vx'] or ball_vy != self.state_variables['ball_vy']:
            # velocity will change for 2 consecutive frames, but only register the first frame
            if not self.state_variables['collision']:
                collision = True

        self.state_variables['lives'] = lives
        self.state_variables['ball_x'] = ball_x
        self.state_variables['ball_y'] = ball_y
        self.state_variables['paddle_x'] = paddle_x
        self.state_variables['ball_vx'] = ball_vx
        self.state_variables['ball_vy'] = ball_vy
        self.state_variables['collision'] = collision
        self.state_variables['brick_hit'] = self.reward > 0
        self.state_variables['bricks_hit'] += self.reward > 0
        self.state_variables['losing life'] = losing_life

        self.step_counter += 1
        # if measurements are weird then ball just spawned or is unfired
        weird = False
        if ball_x < 0 or ball_y < 0:
            weird = True
        if ball_vx == 0 or ball_vy == 0:
            weird = True
        if abs(ball_vx) > 10 or abs(ball_vy) > 10:
            weird = True

        if not weird:
            # to get more special data save all collisions
            if self.step_counter % self.save_interval == 0 or self.state_variables['collision'] or losing_life:
                self.game_steps.append(GameStep(self.observation, self.env.render(), self.state_variables))

        # for debugging
        self.env.state_variables = self.state_variables

        return self.observation, self.reward, self.termination, self.truncation, self.info

    def save_data(self):
        print(f"Filtering {len(self.game_steps)} game steps...")
        unique_game_steps = []
        seen = set()

        for game_step in self.game_steps:
            # Create a frozenset of the state variables
            state_identifier = frozenset(game_step.state_variables.items())
            if state_identifier not in seen:
                unique_game_steps.append(game_step)
                seen.add(state_identifier)

        print(f"Saving {len(unique_game_steps)} unique game steps")
        prepare_folders(f"../data")
        with open('../data/game_steps.pickle', 'wb') as f:
            pickle.dump(unique_game_steps, f)