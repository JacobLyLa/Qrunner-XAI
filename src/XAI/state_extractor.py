import pickle
import random

import gymnasium as gym
import numpy as np


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

class EnvStep:
    def __init__(self, observation, image, state_variables):
        self.observation = np.array(observation).copy()
        # When render mode is human, image is None
        if image is not None: 
            self.image = image.copy()
        self.state_variables = state_variables.copy()

class StateExtractorWrapper(gym.Wrapper):
    def __init__(self, env, save_interval):
        gym.Wrapper.__init__(self, env)
        self.env.state_variables = None
        self.env_steps = []
        self.total_steps = 0
        self.save_interval = save_interval

    def step(self, action):
        self.observation, self.reward, self.termination, self.truncation, self.info = self.env.step(action)

        # Update state variables
        self.state_variables['player jumping'] = self.env.unwrapped.player.jumping
        self.state_variables['player falling'] = self.env.unwrapped.player.gravity_count > 0
        self.state_variables['player dodging'] = self.env.unwrapped.player.dodging
        self.state_variables['player y velocity'] = self.env.unwrapped.player.gravity_count
        self.state_variables['player on wall'] = self.env.unwrapped.player.standing_on != None
        self.state_variables['player x position'] = self.env.unwrapped.player.x
        self.state_variables['player on grass'] = self.env.unwrapped.player.y == 58
        self.state_variables['player y position'] = self.env.unwrapped.player.y

        # Can't save lists in state variables, so we create all necessary extractions from the list here
        events = self.env.unwrapped.active_events
        camera_offset = self.env.unwrapped.camera_offset_x
        size = self.env.unwrapped.GAME_SIZE
        
        # Events within the camera view
        visible_events = [e for e in events if e.x - camera_offset < size and e.x + e.width - camera_offset > 0]
        self.state_variables['visible events'] = len(visible_events)
        self.state_variables['visible walls'] = len([e for e in visible_events if type(e).__name__ == 'Wall'])
        self.state_variables['visible air walls'] = len([e for e in visible_events if type(e).__name__ == 'Wall' and e.wall_type == 'air'])
        self.state_variables['visible bullets'] = len([e for e in visible_events if type(e).__name__ == 'Bullet'])
        self.state_variables['visible lava'] = len([e for e in visible_events if type(e).__name__ == 'Lava'])
        self.state_variables['visible blue coins'] = len([e for e in visible_events if type(e).__name__ == 'Coin' and e.coin_type == 'blue'])
        self.state_variables['visible gold coins'] = len([e for e in visible_events if type(e).__name__ == 'Coin' and e.coin_type == 'gold'])
        self.state_variables['visible red coins'] = len([e for e in visible_events if type(e).__name__ == 'Coin' and e.coin_type == 'red'])
        self.state_variables['visible good events'] = self.state_variables['visible blue coins'] + self.state_variables['visible gold coins']
        self.state_variables['visible bad events'] = self.state_variables['visible red coins'] + self.state_variables['visible lava'] + self.state_variables['visible bullets']
        self.state_variables['visible ghosts'] = len([e for e in visible_events if type(e).__name__ == 'Ghost'])
        self.state_variables['visible ghost'] = self.state_variables['visible ghosts'] > 0

        # Calculate wall area
        self.state_variables['total wall area'] = 0
        for event in visible_events:
            if type(event).__name__ == 'Wall':
                self.state_variables['total wall area'] += event.width * event.height

        # Bullet close
        self.state_variables['bullet close'] = 0
        for event in visible_events:
            if type(event).__name__ == 'Bullet':
                self.state_variables['bullet close'] = max(self.state_variables['bullet close'], 
                                                                     84 - euclidean_distance(event.x, event.y, 
                                                                                         self.state_variables['player x position'], 
                                                                                         self.state_variables['player y position']))
                
        # Bullet aligned with player (bullet y between player y and player y + player height)
        self.state_variables['bullet aligned with player'] = False
        for event in visible_events:
            if type(event).__name__ == 'Bullet':
                if event.y >= self.state_variables['player y position'] and event.y <= self.state_variables['player y position'] + self.env.unwrapped.player.height:
                    # Bullet also to the right of player
                    if event.x > self.state_variables['player x position']:
                        self.state_variables['bullet aligned with player'] = True
                        break
                    
        # Bullet above player (overlap in x and )
        self.state_variables['bullet above player'] = False
        for event in visible_events:
            if type(event).__name__ == 'Bullet':
                if self.state_variables['player y position'] - event.y < 20:
                    if event.x + event.width > self.state_variables['player x position'] and event.x < self.state_variables['player x position'] + self.env.unwrapped.player.width:
                        self.state_variables['bullet above player'] = True
                        break
                    
        # Wall within 20 units to the right of player
        self.state_variables['wall right of player'] = False
        if not self.state_variables['visible bullets']:
            for event in visible_events:
                if type(event).__name__ == 'Wall':
                    if event.wall_type == 'ground':
                        if event.x > self.state_variables['player x position']:
                            if event.x < self.state_variables['player x position'] + 20:
                                self.state_variables['wall right of player'] = True
                                break
                    
        # Good coin left of player
        self.state_variables['good coin left of player'] = False
        for event in visible_events:
            if type(event).__name__ == 'Coin':
                if event.coin_type == 'gold' or event.coin_type == 'blue':
                    if event.x < self.state_variables['player x position']:
                        self.state_variables['good coin left of player'] = True
                        break
                    
        # Unreachable and Reachable good coin
        self.state_variables['visible high coin'] = False
        self.state_variables['visible good low coin'] = False
        for event in visible_events:
            if type(event).__name__ == 'Coin':
                if event.coin_type == 'gold' or event.coin_type == 'blue':
                    if event.y < 42:
                        self.state_variables['visible high coin'] = True
                    else:
                        self.state_variables['visible good low coin'] = True
                        
        self.total_steps += 1
        
        # If there are some special states, then save anyway
        special = False
        special_states = ['bullet aligned with player', 'player dodging', 'good coin left of player', 'bullet above player', 'wall right of player']
        if any([self.state_variables[state] for state in special_states]):
            special = True        
    
        # Check if this step will be saved
        if self.total_steps % self.save_interval == 0 or special:
            self.env_steps.append(EnvStep(self.observation, self.env.render(), self.state_variables))
        
        return self.observation, self.reward, self.termination, self.truncation, self.info
    
    def reset(self, **kwargs):
        self.observation = self.env.reset(**kwargs)
        self.state_variables = {}
        return self.observation

    def save_data(self):
        print(f"Filtering {len(self.env_steps)} game steps...")
        unique_env_steps = []
        seen = set()

        for game_step in self.env_steps:
            # Create a frozenset of the state variables
            state_identifier = frozenset(game_step.state_variables.items())
            if state_identifier not in seen:
                unique_env_steps.append(game_step)
                seen.add(state_identifier)
                
        # If too many steps, sample self.total_steps//self.save_interval steps
        to_save = min(len(unique_env_steps), self.total_steps//self.save_interval)
        unique_env_steps = random.sample(unique_env_steps, to_save)

        print(f"Saving {len(unique_env_steps)} unique game steps")
        with open('./data/env_steps.pickle', 'wb') as f:
            pickle.dump(unique_env_steps, f)