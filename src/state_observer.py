import gymnasium as gym

class StateObserverWrapper(gym.Wrapper):
    ram_map = {
        'ball_x': 99,
        'ball_y': 101,
        'player_x': 72,
        'bricks_hit_count': 77,
        'bricks_map': range(30), # TODO: figure out mapping
        'score': 84,
    }
    def __init__(self, env: gym.Env):
        gym.Wrapper.__init__(self, env)
        self.concepts = []

        # env variables
        self.observation = None
        self.reward = None
        self.termination = None
        self.truncation = None
        self.info = None
        self.image = None

        # collection of state variables
        self.state_variables = {
            'lives': 5,
            'ball_x': 68, # initial position
            'ball_y': 116, # initial position
            'player_x': None,
            'bricks_hit_count': None,
            'bricks_map': None,
            'score': None,
            'ball_vx': 4, # initial velocity
            'ball_vy': 4, # initial velocity
            'collision': False,
            'lost_life': False,
        }

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        # reset env and also this entire wrapper (TODO: not needed?)
        concepts = self.concepts # remember concepts
        self = StateObserverWrapper(self.env) # reset self
        self.add_concepts(concepts) # add concepts again
        return obs, info

    def step(self, action):
        self.observation, self.reward, self.termination, self.truncation, self.info = self.env.step(action)
        self.update_state()
        return self.observation, self.reward, self.termination, self.truncation, self.info

    def update_state(self):
        # image can be directly read from env
        self.image = self.env.render()

        # read state variables from RAM
        ram = self.env.unwrapped.ale.getRAM()
        ram_map = StateObserverWrapper.ram_map
        new_ball_x = int(ram[ram_map['ball_x']])
        new_ball_y = int(ram[ram_map['ball_y']])
        if new_ball_x == 0 and new_ball_y == 0: # ball hasn't been fired
            # TODO: but maybe a concept should be that the ball hasn't been fired yet?
            return

        brick_map = ram[ram_map['bricks_map']]
        self.state_variables['bricks_map'] = ''.join(format(byte, '08b') for byte in brick_map)
        self.state_variables['player_x'] = int(ram[ram_map['player_x']]) + 8
        self.state_variables['bricks_hit_count'] = int(ram[ram_map['bricks_hit_count']])
        self.state_variables['score'] = int(ram[ram_map['score']])
        new_lives = int(self.env.unwrapped.ale.lives())
        if new_lives < self.state_variables['lives']:
            self.state_variables['lost_life'] = True
        else:
            self.state_variables['lost_life'] = False
        self.state_variables['lives'] = new_lives

        new_ball_vx = new_ball_x - self.state_variables['ball_x']
        new_ball_vy = new_ball_y - self.state_variables['ball_y']
        self.state_variables['ball_x'] = new_ball_x
        self.state_variables['ball_y'] = new_ball_y
        # if velocity changes, then there was a collision.
        # but velocity will also change an additional time after a collision
        if (new_ball_vx != self.state_variables['ball_vx'] or
            new_ball_vy != self.state_variables['ball_vy']) and \
            not self.state_variables['collision']:
            self.state_variables['collision'] = True
        else:
            self.state_variables['collision'] = False
        self.state_variables['ball_vx'] = new_ball_vx
        self.state_variables['ball_vy'] = new_ball_vy

        self.check_for_concepts()

    def check_for_concepts(self):
        for concept in self.concepts:
            concept.check_observation(self.observation, self.state_variables, self.image)

    def save_concepts(self):
        for concept in self.concepts:
            concept.save_observations()

    def add_concepts(self, concepts):
        self.concepts.extend(concepts)
