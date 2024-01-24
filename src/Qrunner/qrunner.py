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
    def __init__(self, size=252):
        super(QrunnerEnv, self).__init__()
        pygame.init()
        # Game constants
        self.size = size
        self.ground_height = self.size // 10
        self.frames_since_reward_limit = 300 # TODO: move to wrapper
        self.reward_per_pixel = 1 / (2 * self.size) # so 1 reward per 2 screen
        self.start_event_frequency = 0.5 * self.size # every 0.5 screen
        self.end_event_frequency = 0.1 * self.size # every 0.1 screen
        self.event_frequency_update = self.size // 5 # everytime the camera offset increases by this, the event frequency is += 1
        self.camera_lock_x = 0.3 * self.size

        # Player constants
        self.player_width = self.size // 12
        self.player_height = self.size // 6
        self.frames_in_air = 30
        self.velocity_x = 0.02 * self.size
        self.gravity = 0.002 * self.size
        
        # Pygame stuff
        self.game_over_font = pygame.font.Font('freesansbold.ttf', self.size // 12)
        self.normal_font = pygame.font.Font('freesansbold.ttf', int(0.5*self.player_width))
        self.grass_color = (135, 206, 250)
        # self.grass_image = pygame.image.load("Qrunner/resources/grass_sprite.jpg")
        # self.available_events = [Coin, Wall, Bullet, Lava, Star, Portal, Shuriken]
        self.available_events = [Bullet, Coin, Lava, Wall]
        self.event_weights = [event.WEIGHT for event in self.available_events]
        
        # Calculate velocity y such that the player can jump for the specified number of frames
        i = 0
        self.velocity_y = self.gravity
        while self.calculate_frames_in_air() <= self.frames_in_air and i < 100:
            i += 1
            self.velocity_y = self.gravity * (1 + i * 0.5)
        # print(f"Velocity y: {self.velocity_y}")
        # print(f"Actual frames in air: {self.calculate_frames_in_air()}")

        # Gymnasium variables
        # Only rgb_array supported, use run for human mode
        self.render_mode = "rgb_array" 
        # 0: noop, 1: left, 2: right, 3: jump, 4 dodge down
        self.num_actions = 5
        self.action_space = spaces.Discrete(self.num_actions)
        # channel first or last? looks like openCV wants last
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.size, self.size, 3), dtype=np.uint8)
        self.reward_range = (-1, 1)
        self.screen = pygame.Surface((size, size))
        
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
        
        self.player = Player(self.camera_lock_x, self.size - self.ground_height - self.player_height, self.player_width, self.player_height, self.velocity_x, self.velocity_y)
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
        self.screen.fill(self.grass_color)
        self.draw_grass()

        # Draw events
        for event in self.active_events:
            if not isinstance(event, Shuriken):  # Draw all events except shurikens
                event.draw(self.screen, self.camera_offset_x)
        for event in self.active_events:
            if isinstance(event, Shuriken):
                event.draw(self.screen, self.camera_offset_x)

        # Draw player
        self.player.draw(self.screen, self.camera_offset_x)

        # Draw number of events
        text = self.normal_font.render(f'Events: {len(self.active_events)}', True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.topright = (self.size, 0)
        self.screen.blit(text, textRect)

        pixel_data = pygame.surfarray.array3d(self.screen)
        observation = np.transpose(pixel_data, (1, 0, 2))
        self.last_observation = observation
        return observation
    
    def _render_to_screen(self):
        # Get the main display surface from Pygame
        display_surface = pygame.display.get_surface()

        # Blit (copy) the contents of self.screen onto the display_surface
        display_surface.blit(self.screen, (0, 0))

        # Update the full display Surface to the screen
        pygame.display.update()

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
                self.player.score += self.reward_per_pixel * (self.player.x - self.last_reward)
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
        if self.player.y + self.player.height > self.size - self.ground_height:
            self.player.y = self.size - self.ground_height - self.player.height
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
        if self.last_observation is not None:
            # Return a copy of the last observation
            return np.copy(self.last_observation)
        else:
            assert False, "No observation to render"

    def close(self):
        pygame.display.quit()
        self.screen = None
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
        pygame.draw.rect(self.screen, (0, 255, 0), (0, self.size - self.ground_height, self.size, self.ground_height))
        '''
        grass_width = self.grass_image.get_width()

        # Calculate the start position of the first grass image based on the camera offset
        start_x = -(self.camera_offset_x % grass_width)

        # Adjust start_x to always begin within the screen area
        if start_x > 0:
            start_x -= grass_width

        # Tile the grass image
        grass_x = start_x
        while grass_x < self.size:
            screen.blit(self.grass_image, (grass_x, self.size - self.ground_height))
            grass_x += grass_width
        '''

    # Human mode, clock ticks, render frames to pygame screen, freeze at game over, R for reset
    def run(self):
        pygame.display.set_caption("Qrunner")
        self.display_surface = pygame.display.set_mode((self.size, self.size))
        clock = pygame.time.Clock()
        run = True
        while run:
            clock.tick(60)

            # Check for exit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

            # Check for reset game            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_r]:
                self.reset()

            # Freeze on game over
            if self.game_over:
                text = self.game_over_font.render(f'Game Over', True, (0, 0, 0))
                textRect = text.get_rect()
                textRect.center = (self.size // 2, self.size // 2)
                self.screen.blit(text, textRect)
                self._render_to_screen()
                #pygame.display.update()
                continue

            # Act on key presses
            '''
            Currently only one key can be pressed at a time.
            This is to simplify the action space for the RL agent.
            Can allow multiple for humans, but restrict for RL agent.
            But then benchmarks are not that comparable.
            '''

            # Check if any key is pressed.
            # Note jump is prioritized over movement etc.
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
            self._render_to_screen()

        pygame.quit()

if __name__ == "__main__":
    game = QrunnerEnv(size=84*6)
    game.reset()
    game.run()