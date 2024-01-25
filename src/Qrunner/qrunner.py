import random
import time

import cv2
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from src.Qrunner.event import Bullet, Coin, Lava, Portal, Shuriken, Star, Wall
from src.Qrunner.player import Player

class QrunnerEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 60}
    GAME_SIZE = 84
    def __init__(self):
        super(QrunnerEnv, self).__init__()
        pygame.init()
        # Game constants
        self.ground_height = self.GAME_SIZE // 10
        
        self.frames_since_reward_limit = 300 # TODO: move to wrapper
        
        self.reward_per_screen = 0.5
        self.start_event_frequency = self.GAME_SIZE // 2
        self.end_event_frequency = self.GAME_SIZE // 10
        self.event_frequency_update = self.GAME_SIZE
        self.camera_lock_x = self.GAME_SIZE // 3
        
        # Player constants
        self.frames_in_air = 30
        self.player_width = 8
        self.player_height = 18
        self.velocity_x = 2
        self.gravity = 0.2
        
        # Pygame stuff
        self.sky_color = (135, 206, 250)
        # self.grass_image = pygame.image.load("Qrunner/resources/grass_sprite.jpg")
        # self.available_events = [Coin, Wall, Bullet, Lava, Star, Portal, Shuriken]
        self.available_events = [Bullet, Coin, Lava, Wall]
        self.event_weights = [event.WEIGHT for event in self.available_events]
        
        # Calculate velocity y such that the player can jump for the specified number of frames
        i = 0
        self.velocity_y = self.gravity
        while self.calculate_frames_in_air() <= self.frames_in_air and i < self.GAME_SIZE:
            i += 1
            self.velocity_y = self.gravity * (1 + i * 0.5)
        print(f"Velocity y: {self.velocity_y}")
        # print(f"Actual frames in air: {self.calculate_frames_in_air()}")

        # Gymnasium variables
        # Only rgb_array supported, use run for human mode
        self.render_mode = "rgb_array" 
        # 0: noop, 1: left, 2: right, 3: jump, 4 dodge down
        self.num_actions = 5
        self.action_space = spaces.Discrete(self.num_actions)
        # channel first or last? looks like openCV wants last
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.GAME_SIZE, self.GAME_SIZE, 3), dtype=np.uint8)
        self.reward_range = (-1, 1)
        
        # Internal surface for game logic (fixed size)
        self.surface = pygame.Surface((self.GAME_SIZE, self.GAME_SIZE))
        
    def calculate_frames_in_air(self):
        frames_in_air = 0
        gravity_count = 0
        y = 0
        while y >= 0:
            gravity_count += self.gravity
            y -= gravity_count
            y += self.velocity_y

            if y > 0:
                frames_in_air += 1

        return frames_in_air
        
    def get_action_meanings(self):
        return ['NOOP', 'LEFT', 'RIGHT', 'JUMP', 'DODGE']

    def reset(self, seed=None, options=None):
        random.seed(seed)
        self.camera_offset_x = 0
        
        self.player = Player(self.camera_lock_x, self.GAME_SIZE - self.ground_height - self.player_height, self.player_width, self.player_height, self.velocity_x, self.velocity_y)
        self.game_over = False
        self.spawned_events = 0
        self.active_events = []

        # Reward for going to the right
        self.frames_since_reward = 0
        self.last_reward = self.camera_lock_x

        # Start game with a random event
        self.last_event_x = 0
        self.generate_event()

        obs = self._generate_observation()
        info = {}
        return obs, info # gym expects 2 values but vec env 1?
    
    def _generate_observation(self):
        # Draw background, grass
        self.surface.fill(self.sky_color)
        self.draw_grass()

        # Draw events
        for event in self.active_events:
            if not isinstance(event, Shuriken):  # Draw all events except shurikens
                event.draw(self.surface, self.camera_offset_x)
        for event in self.active_events:
            if isinstance(event, Shuriken):
                event.draw(self.surface, self.camera_offset_x)

        # Draw player
        self.player.draw(self.surface, self.camera_offset_x)

        # Draw number of events
        '''
        text = self.normal_font.render(f'Events: {len(self.active_events)}', True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.topright = (self.GAME_SIZE, 0)
        self.surface.blit(text, textRect)
        '''
        observation = pygame.surfarray.array3d(self.surface)
        observation = np.transpose(observation, (1, 0, 2))
        self.last_observation = observation
        return observation
    
    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action} not in {self.action_space}"
        
        if action == 0: # Do nothing
            pass
        if action == 1:  # Left
            if self.player.x > self.camera_offset_x:
                self.player.x -= self.player.velocity_x
        elif action == 2:  # Right
            self.player.x += self.player.velocity_x
            if self.need_to_generate_event():
                self.generate_event()
            if self.player.x - self.last_reward >= 1:
                self.player.score += self.reward_per_screen * (self.player.x - self.last_reward) / self.GAME_SIZE
                self.last_reward = self.player.x
        elif action == 3:  # Jump
            if self.player.gravity_count == 0:
                self.player.jumping = True
                self.player.gravity_count = self.gravity
        elif action == 4:  # Dodge down
            if not self.player.dodging:
                self.player.y += self.player.height // 2
                self.player.height //= 2
                self.player.dodging = True
        if action != 4 and self.player.dodging: # Stop dodging
            self.player.y -= self.player.height
            self.player.height *= 2
            self.player.dodging = False

        # Update the game state
        if self.player.gravity_count > 0:
            # Jumping
            if self.player.jumping:
                self.player.y -= self.player.velocity_y
            # Falling
            self.player.y += self.player.gravity_count
            self.player.gravity_count += self.gravity

        # Prevent player from falling below ground
        if self.player.y + self.player.height > self.GAME_SIZE - self.ground_height:
            self.player.y = self.GAME_SIZE - self.ground_height - self.player.height
            self.player.gravity_count = 0
            self.player.jumping = False

        to_remove = []
        # Update all wall events first
        for event in self.active_events:
            if isinstance(event, Wall):
                if event.frame_update_remove():
                    to_remove.append(event)
                
        # Then update all other events
        for event in self.active_events:
            if not isinstance(event, Wall):
                if event.frame_update_remove():
                    to_remove.append(event)
                    
        for event in to_remove:
            self.active_events.remove(event)
            
        # update x_prev and y_prev
        self.player.x_prev = self.player.x
        self.player.y_prev = self.player.y
            
        # Update camera offset
        if self.player.x - self.camera_offset_x > self.camera_lock_x:
            self.camera_offset_x = self.player.x - self.camera_lock_x

        # Calculate possible reward and check for time limit
        reward = self.player.score - self.player.score_prev
        self.player.score_prev = self.player.score
        # Also check if env should be truncated because of time limit
        truncated = False
        if reward != 0:
            self.player.frames_since_reward = 0
        else:
            self.player.frames_since_reward += 1
            if self.player.frames_since_reward > self.frames_since_reward_limit:
                truncated = True

        # Create next observation
        observation = self._generate_observation()

        # Additional info?
        info = {}
        terminated = self.game_over

        return observation, reward, terminated, truncated, info

    def render(self):
        return np.copy(self.last_observation)

    def close(self):
        pygame.display.quit()
        self.surface = None
        pygame.quit()

    def need_to_generate_event(self):
        event_frequency = int(max(self.end_event_frequency, self.start_event_frequency - (self.camera_offset_x // self.event_frequency_update)))
        if self.player.x - self.last_event_x > event_frequency:
            return True
        return False

    def generate_event(self):
        self.last_event_x = self.player.x
        event = random.choices(self.available_events, weights=self.event_weights, k=1)[0]
        event = event(self)
        if not event.fail:
            self.active_events.append(event)

    def draw_grass(self):
        pygame.draw.rect(self.surface, (0, 255, 0), (0, self.GAME_SIZE - self.ground_height, self.GAME_SIZE, self.ground_height))
        '''
        grass_width = self.grass_image.get_width()

        # Calculate the start position of the first grass image based on the camera offset
        start_x = -(self.camera_offset_x % grass_width)

        # Adjust start_x to always begin within the surface area
        if start_x > 0:
            start_x -= grass_width

        # Tile the grass image
        grass_x = start_x
        while grass_x < self.GAME_SIZE:
            surface.blit(self.grass_image, (grass_x, self.GAME_SIZE - self.ground_height))
            grass_x += grass_width
        '''

    # Human mode, clock ticks, render frames to pygame surface, freeze at game over, R for reset
    def run(self, scale=1):
        screen_size = self.GAME_SIZE * scale
        game_over_font = pygame.font.Font('freesansbold.ttf', screen_size // 8)
        normal_font = pygame.font.Font('freesansbold.ttf', screen_size // 16)
        pygame.display.set_caption("Qrunner")
        display_surface = pygame.display.set_mode((screen_size, screen_size))
        clock = pygame.time.Clock()
        run = True
        game_over = False
        while run:
            clock.tick(60)

            # Check for exit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

            # Check for reset game            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_r]:
                game_over = False
                self.reset()

            # Freeze on game over, only render and print score once
            if self.game_over:
                if game_over:
                    continue
                print(f"Score: {self.player.score:.2f}")
                game_over = True
                text = game_over_font.render(f'Game Over', True, (0, 0, 0))
                textRect = text.get_rect()
                textRect.center = (screen_size // 2, screen_size // 2)
                display_surface.blit(text, textRect)
                text = normal_font.render(f'Press R to restart', True, (0, 0, 0))
                textRect = text.get_rect()
                textRect.center = (screen_size // 2, screen_size // 2 + 0.1*screen_size)
                display_surface.blit(text, textRect)
                pygame.display.update()
                continue

            # Act on key presses
            # Latter actions override former actions
            action = 0
            if keys[pygame.K_a]:
                action = 1
            if keys[pygame.K_d]:
                action = 2
            if keys[pygame.K_SPACE]:
                action = 3
            if keys[pygame.K_s]:
                action = 4
            observation, reward, terminated, truncated, info = self.step(action)
            surface = pygame.surfarray.make_surface(observation)
            # Scale surface
            surface = pygame.transform.scale(surface, (screen_size, screen_size))
            display_surface.blit(surface, (0, 0))
            pygame.display.update()

        pygame.quit()

if __name__ == "__main__":
    game = QrunnerEnv()
    game.reset()
    game.run(1)