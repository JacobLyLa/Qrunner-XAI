import random

import pygame

from src.Qrunner.event import Coin, Wall, Bullet, Lava
from src.Qrunner.qrunner import QrunnerEnv


class ModifiedQrunnerEnv(QrunnerEnv):
    def __init__(self):
        super(ModifiedQrunnerEnv, self).__init__()
        #self.available_events = [Coin, Wall, Bullet, Lava]  # Only allow coins to spawn
        #self.event_weights = [0.9, 0.06, 0.02, 0.02]
        self.coins_picked = []
        self.coin_values = {'blue': 0.2, 'red': -0.2, 'gold': 1}

    '''
    def need_to_generate_event(self):
        return len(self.active_events) < 4
    
    def generate_event(self):
        event = random.choices(self.available_events, weights=self.event_weights, k=1)[0]
        event = event(self)
        if isinstance(event, Coin):
            if random.random() < 0.5:
                event.x = self.camera_offset_x + random.randint(5, 84)
                event.y = random.randint(40,70)
        if not event.fail:
            self.active_events.append(event)
    '''    
    
    def step(self, action):

        
        observation, reward, terminated, truncated, info = super(ModifiedQrunnerEnv, self).step(action)
        if len(self.player.coins_picked) == 0: # Env reset
            self.coins_picked = []
        if len(self.player.coins_picked) > len(self.coins_picked):
            new_coins = self.player.coins_picked[len(self.coins_picked):]
            reward = sum([self.coin_values[coin] for coin in new_coins])
            self.coins_picked = self.player.coins_picked.copy()
            if reward > 0:
                coin = Coin(self)
                coin.x = self.player.x - random.randint(2, 25)
                coin.y = random.randint(35,70)
                self.active_events.append(coin)
        else:
            reward = 0
        
        return observation, reward, terminated, truncated, info

if __name__ == "__main__":
    game = ModifiedQrunnerEnv()
    game.reset()
    game.run(scale=8, evaluate=-1)
