import math
import random

from concept import Concept


def frames_until_collision_vertical(state_variables):
    ball_y = state_variables['ball_y']
    ball_vy = state_variables['ball_vy']
    if ball_vy == 0:
        print("ball_vy is 0")
        return 0
    if ball_vy > 0: # check when it hits paddle @ 185
        return max((185-ball_y), 0) / ball_vy
    else: # check when it hits top (bottom row of bricks) @ 100
        return max((ball_y-100), 0) / abs(ball_vy)


def frames_until_collision_horizontal(state_variables):
    ball_x = state_variables['ball_x']
    ball_vx = state_variables['ball_vx']
    if ball_vx == 0:
        print("ball_vx is 0")
        return 0
    if ball_vx > 0: # check when it hits right wall @ 149
        return max((149-ball_x), 0) / ball_vx
    else: # check when it hits left wall @ 11
        return max((ball_x-11), 0) / abs(ball_vx)

concept_instances = [
    Concept(name="random (b)", binary=True,
    value_function=lambda state_variables: random.random() < 0.5
    ),

    Concept(name="all lives (b)", binary=True,
    value_function=lambda state_variables: 
        state_variables['lives'] == 5
    ),

    Concept(name="last life (b)", binary=True,
    value_function=lambda state_variables: 
        state_variables['lives'] == 1
    ),

    Concept(name="reward (b)", binary=True,
    value_function=lambda state_variables: 
        state_variables['reward'],
    ),

    Concept(name="ball collision (b)", binary=True,
    value_function=lambda state_variables: 
        state_variables['collision'],
    ),

    Concept(name="ball low (b)", binary=True,
    value_function=lambda state_variables: 
        state_variables['ball_y'] > 150,
    ),

    Concept(name="ball left paddle (b)", binary=True,
    value_function=lambda state_variables: 
        state_variables['ball_x'] < state_variables['paddle_x'],
    ),

    Concept(name="ball right paddle (b)", binary=True,
    value_function=lambda state_variables: 
        state_variables['ball_x'] > state_variables['paddle_x'],
    ),

    Concept(name="ball distance paddle", binary=False,
    value_function=lambda state_variables: 
        math.sqrt(
            (state_variables['paddle_x'] - state_variables['ball_x'])**2 + 
            (190 - state_variables['ball_y'])**2),
    ),

    Concept(name="ball y", binary=False,
    value_function=lambda state_variables: 
        state_variables['ball_y'],
    ),

    Concept(name="ball y next", binary=False,
    value_function=lambda state_variables: 
        (state_variables)['ball_y'] + state_variables['ball_vy'],
    ),

    Concept(name="ball x", binary=False,
    value_function=lambda state_variables: 
        state_variables['ball_x'],
    ),

    Concept(name="ball x next", binary=False,
    value_function=lambda state_variables: 
        (state_variables['ball_x'] + state_variables['ball_vx']),
    ),

    Concept(name="lives", binary=False,
    value_function=lambda state_variables: 
        state_variables['lives'],
    ),

    Concept(name="x diff", binary=False,
    value_function=lambda state_variables:
        abs(state_variables['paddle_x'] - state_variables['ball_x']),
    ),

    Concept(name="frames until collision horizontal", binary=False,
    value_function=frames_until_collision_horizontal,
    ),

    Concept(name="frames until collision vertical", binary=False,
    value_function=frames_until_collision_vertical,
    ),

    Concept(name="paddle x", binary=False,
    value_function=lambda state_variables: 
        state_variables['paddle_x'],
    ),

    Concept(name="ball speed horizontal", binary=False,
    value_function=lambda state_variables: 
        abs(state_variables['ball_vx']),
    ),

    Concept(name="ball speed vertical", binary=False,
    value_function=lambda state_variables:
        abs(state_variables['ball_vy']),
    ),

    Concept(name="game steps", binary=False,
    value_function=lambda state_variables:
        state_variables['game_steps'],
    ),

    Concept(name="bricks hit", binary=False,
    value_function=lambda state_variables:
        state_variables['bricks_hit'],
    ),
]

concept_instances = {concept.name: concept for concept in concept_instances}