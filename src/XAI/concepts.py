import math
import random

from concept import Concept

concept_instances = [
    # Ball Concepts
    Concept(name="ball y", binary=False,
            value_function=lambda state_variables:
            state_variables['ball_y']),

    Concept(name="ball x", binary=False,
            value_function=lambda state_variables:
            state_variables['ball_x']),

    Concept(name="ball low (b)", binary=True,
            value_function=lambda state_variables:
            state_variables['ball_y'] > 150),

    Concept(name="ball collision (b)", binary=True,
            value_function=lambda state_variables:
            state_variables['collision']),

    Concept(name="ball going right (b)", binary=True,
            value_function=lambda state_variables:
            state_variables['ball_vx'] > 0),

    Concept(name="ball going left (b)", binary=True,
            value_function=lambda state_variables:
            state_variables['ball_vx'] < 0),

    Concept(name="ball going up (b)", binary=True,
            value_function=lambda state_variables:
            state_variables['ball_vy'] < 0),

    Concept(name="ball going down (b)", binary=True,
            value_function=lambda state_variables:
            state_variables['ball_vy'] > 0),

    Concept(name="ball speed", binary=False,
            value_function=lambda state_variables:
            math.sqrt(state_variables['ball_vx']**2 + state_variables['ball_vy']**2)),

    # Paddle Concepts
    Concept(name="paddle x", binary=False,
            value_function=lambda state_variables:
            state_variables['paddle_x']),

    Concept(name="ball left for paddle (b)", binary=True,
            value_function=lambda state_variables:
            state_variables['ball_x'] < state_variables['paddle_x']),

    Concept(name="ball right for paddle (b)", binary=True,
            value_function=lambda state_variables:
            state_variables['ball_x'] > state_variables['paddle_x']),

    Concept(name="ball above paddle (b)", binary=True,
            value_function=lambda state_variables:
            abs(state_variables['ball_x'] - state_variables['paddle_x']) < 10),

    Concept(name="ball paddle distance", binary=False,
            value_function=lambda state_variables:
            math.sqrt((state_variables['paddle_x'] - state_variables['ball_x'])**2 +
                      (190 - state_variables['ball_y'])**2)),

    # Life Related Concepts
    Concept(name="lives", binary=False,
            value_function=lambda state_variables:
            state_variables['lives']),

    Concept(name="last life (b)", binary=True,
            value_function=lambda state_variables:
            state_variables['lives'] == 1),

    Concept(name="losing life (b)", binary=True,
            value_function=lambda state_variables:
            state_variables['losing life'] == 1),

    # Other Concepts
    Concept(name="random (b)", binary=True,
            value_function=lambda state_variables:
            random.random() < 0.5),

    Concept(name="brick hit (b)", binary=True,
            value_function=lambda state_variables:
            state_variables['brick_hit']),

    Concept(name="bricks hit", binary=False,
            value_function=lambda state_variables:
            state_variables['bricks_hit']),
]
concept_instances = {concept.name: concept for concept in concept_instances}