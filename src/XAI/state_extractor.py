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
        self.state_variables['player fall velocity'] = self.env.unwrapped.player.gravity_count
        self.state_variables['player standing on wall'] = self.env.unwrapped.player.standing_on != None
        self.state_variables['player x position'] = self.env.unwrapped.player.x
        if not self.state_variables['player dodging']:
            self.state_variables['player y position'] = self.env.unwrapped.player.y
        else:
            self.state_variables['player y position'] = self.env.unwrapped.player.y - self.env.unwrapped.player.height
        
        self.state_variables['camera x offset'] = self.env.unwrapped.camera_offset_x

        # Can't save lists in state variables, so we create all necessary extractions from the list here
        events = self.env.unwrapped.active_events
        camera_offset = self.env.unwrapped.camera_offset_x
        size = self.env.unwrapped.GAME_SIZE
        
        # Events within the camera view
        visible_events = [e for e in events if e.x - camera_offset < size and e.x + e.width - camera_offset > 0]
        self.state_variables['visible events'] = len(visible_events)
        # Number of walls
        self.state_variables['visible walls'] = len([e for e in visible_events if type(e).__name__ == 'Wall'])
        # Number of bullets
        self.state_variables['visible bullets'] = len([e for e in visible_events if type(e).__name__ == 'Bullet'])
        # Number of lava
        self.state_variables['visible lava'] = len([e for e in visible_events if type(e).__name__ == 'Lava'])
        # Number of coins
        self.state_variables['visible coins'] = len([e for e in visible_events if type(e).__name__ == 'Coin'])
        # Number of good events (coins)
        self.state_variables['good visible events'] = len([e for e in visible_events if hasattr(e, 'good') and e.good])
        # Number of bad events (lava, bullets, red coins)
        self.state_variables['bad visible events'] = len([e for e in visible_events if hasattr(e, 'good') and not e.good]) + self.state_variables['visible lava'] + self.state_variables['visible bullets']
        # Two close bullets
        distance_threshold = 20
        self.state_variables['two close bullets'] = False
        for i in range(len(visible_events)):
            for j in range(i + 1, len(visible_events)):
                if type(visible_events[i]).__name__ == 'Bullet' and type(visible_events[j]).__name__ == 'Bullet':
                    if euclidean_distance(visible_events[i].x, visible_events[i].y, visible_events[j].x, visible_events[j].y) < distance_threshold:
                        self.state_variables['two close bullets'] = True
                        break
        
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
                    self.state_variables['bullet aligned with player'] = True
                    break
                
        # Coin above lava
        self.state_variables['coin above lava'] = False
        for event in visible_events:
            if type(event).__name__ == 'Coin':
                for e in visible_events:
                    if type(e).__name__ == 'Lava' and e.x < event.x and e.x + e.width > event.x:
                        self.state_variables['coin above lava'] = True
                        break
                    
        # Lava below player
        self.state_variables['lava below player'] = False
        for event in visible_events:
            if type(event).__name__ == 'Lava':
                if event.x < self.state_variables['player x position'] and event.x + event.width > self.state_variables['player x position']:
                    self.state_variables['lava below player'] = True
                    break
                
        # Bullet below player
        self.state_variables['bullet below player'] = False
        for event in visible_events:
            if type(event).__name__ == 'Bullet':
                if self.env.unwrapped.player.y + self.env.unwrapped.player.height < event.y + event.height:
                    if event.x < self.state_variables['player x position'] and event.x + event.width > self.state_variables['player x position']:
                        self.state_variables['bullet below player'] = True
                        break
                        
        self.state_variables['episode steps'] += 1
        self.total_steps += 1
        
        # If there are some special states, then save anyway
        special = False
        if self.state_variables['bullet below player'] or self.state_variables['lava below player'] or self.state_variables['two close bullets']:
            special = True        
    
        # Check if this step will be saved
        if self.total_steps % self.save_interval == 0 or special:
            self.env_steps.append(EnvStep(self.observation, self.env.render(), self.state_variables))
        
        return self.observation, self.reward, self.termination, self.truncation, self.info
    
    def reset(self, **kwargs):
        self.observation = self.env.reset(**kwargs)
        self.state_variables = {
            'episode steps': 0,
            'player jumping': False,
            'player falling': False,
        }
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