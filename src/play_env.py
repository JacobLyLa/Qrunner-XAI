import random
import sys

import gymnasium as gym
import numpy as np
from gymnasium.utils.play import play

from custom_env import make_env
from game_step_saver import GameStepSaverWrapper


def renderOverrider(env):
    env.original_render = env.render
    def render():
        rendered = env.original_render()
        state_variables = env.state_variables
        if state_variables:
            if state_variables['collision']:
                print("Collision:", state_variables['ball_x'], state_variables['ball_y'])
            # print state variables
            string = ""
            for key, value in state_variables.items():
                if key == 'bricks_map':
                    continue
                string += f"{key}: {value} "
            sys.stdout.write(f"\r{string}")
            sys.stdout.flush()

            # draw ball
            ball_x = state_variables['ball_x']
            ball_y = state_variables['ball_y']
            if ball_x > 0 and ball_y > 0:
                if ball_x < rendered.shape[1] and ball_y < rendered.shape[0]:
                    rendered[ball_y, ball_x, 0] = 0
                    rendered[ball_y, ball_x, 1] = 0
                    rendered[ball_y, ball_x, 2] = 255

            # draw player
            paddle_x = state_variables['paddle_x']
            player_y = 190
            if paddle_x > 0:
                if paddle_x < rendered.shape[1]:
                    rendered[player_y, paddle_x, 0] = 0
                    rendered[player_y, paddle_x, 1] = 255
                    rendered[player_y, paddle_x, 2] = 0
        return rendered

    return render

if __name__ == '__main__':
    env = make_env(0, save_interval=99999999)
    # env.render = renderOverrider(env)
    play(env, fps=30, zoom=3)
    env.close()