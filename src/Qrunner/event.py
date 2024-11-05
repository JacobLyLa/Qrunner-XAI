import math
import random
from abc import ABC, abstractmethod

import pygame

# TODO:
# if event is outside of screen skip drawing it

# TODO:
# falls of if at the very tip of a wall
# if player stands on wall then goes off then it can get backup by going towards it (only when falling not jumping)

# TODO:
# when checking for too much overlap it might not work to just randomly generate a new event.
# this can infinitely loop. instead just generate no event?
class Event(ABC):
    def __init__(self, game):
        self.game = game
        self.fail = False

    def overlap(self, player):
        if (player.x + player.width < self.x or  # Player's right side is left of Wall's left side
            player.x > self.x + self.width or    # Player's left side is right of Wall's right side
            player.y > self.y + self.height or   # Player's top side is below Wall's bottom side
            player.y + player.height < self.y):  # Player's bottom side is above Wall's top side
            return False # No overlap
        return True
    
    def overlap_ratio(self, event):
        # Calculate overlap in x-axis
        overlap_x = min(self.x + self.width, event.x + event.width) - max(self.x, event.x)
        if overlap_x < 0: overlap_x = 0  # No overlap in x-axis

        # Calculate overlap in y-axis
        overlap_y = min(self.y + self.height, event.y + event.height) - max(self.y, event.y)
        if overlap_y < 0: overlap_y = 0  # No overlap in y-axis

        overlap_area = overlap_x * overlap_y

        # Calculate areas of the two events
        self_area = self.width * self.height
        event_area = event.width * event.height

        # Use the smaller of the two areas for the ratio
        smaller_area = min(self_area, event_area)

        return overlap_area / smaller_area

    @abstractmethod
    def generate(self, game):
        pass
    
    @abstractmethod
    def frame_update_remove(self):
        pass

class Coin(Event):
    GOLD = (255, 222, 0)
    RED = (255, 50, 25)
    BLUE = (64, 32, 200)
    WEIGHT = 10
    
    def __init__(self, game, x=None, y=None, coin_type=None):
        super().__init__(game)
        self.generate(x, y, coin_type)

    def generate(self, x=None, y=None, coin_type=None):
        s = self.game.GAME_SIZE
        # Decide the type of coin
        if not coin_type:
            coin_type = random.choices(
                ['gold', 'red', 'blue'],
                weights=[20, 20, 60],
                k=1
            )[0]

        if coin_type == 'gold':
            self.color = Coin.GOLD
            self.value = 1
            self.radius = 0.03 * s
            
        elif coin_type == 'blue':
            self.color = Coin.BLUE
            self.value = 0.2
            self.radius = 0.03 * s
            
        elif coin_type == 'red':
            self.color = Coin.RED
            self.value = -0.2
            self.radius = 0.03 * s
            
        self.coin_type = coin_type

        self.width = self.radius * 2
        self.height = self.radius * 2
        if x and y:
            self.x = x - self.radius
            self.y = y - self.radius * 2
        else:
            self.x = self.game.player.x + s + random.randint(0, s // 2)
            self.y = random.randint(s//2, s - self.game.ground_height - s // 10)
            
        # If it's a red coin that we can't walk under or jump over, make it close to ground
        if self.coin_type == 'red':
            player_head_y = self.game.GAME_SIZE - self.game.ground_height - self.game.player_height
            # If it can't be walked under
            if player_head_y < self.y + self.height:
                # If it can't be jumped over
                if player_head_y + self.game.player_height//2 > self.y + self.height:
                    self.y = (self.game.GAME_SIZE - self.game.ground_height) + self.game.player_height // 2
            
        # Check if coin overlaps with wall
        for event in self.game.active_events:
            if isinstance(event, Wall) and event.overlap_ratio(self) > 0:
                self.fail = True
                return

    def frame_update_remove(self):
        if self.x < self.game.camera_offset_x - self.width:
            return True
        if self.overlap(self.game.player):
            self.game.player.score += self.value
            self.game.interactions.append(self.coin_type + ' coin')
            return True
        return False

    def draw(self, window, offset):
        # TODO: Dont draw if its too far right of the screen
        pygame.draw.circle(window, self.color, (self.x - offset + self.radius, self.y + self.radius), self.radius)

class Wall(Event):
    GREY = (128, 128, 128)
    GREY2 = (100, 100, 100)
    WEIGHT = 15
    
    def __init__(self, game):
        super().__init__(game)
        self.generate()

    def generate(self):
        s = self.game.GAME_SIZE
        # Ground wall
        if random.random() < 0.8:
            self.width = random.randint(int(0.2*s), int(0.6*s))
            self.height = random.randint(int(0.1*s), int(0.25*s))
            self.x = self.game.player.x + s + random.randint(0, s // 2)
            self.y = s - self.game.ground_height - self.height
            # Check if wall is covered by another event
            for event in self.game.active_events:
                if event.overlap_ratio(self) > 0.3:
                    self.fail = True
                    return
            self.wall_type = 'ground'
        # Air wall
        # TODO: no collision when jumping from below to avoid impossible levels
        else:
            self.width = random.randint(int(0.2*s), int(0.4*s))
            self.height = random.randint(int(0.03*s), int(0.04*s))
            self.x = self.game.player.x + s + random.randint(0, s // 2)
            self.y = random.randint(int(0.3*s), int(0.5*s))
            # Check if wall is covered by another event
            for event in self.game.active_events:
                if isinstance(event, Wall) and event.overlap_ratio(self) > 0.0:
                    self.fail = True
                    return
            self.wall_type = 'air'
            # Also generate a blue coin on top of it
            coin = Coin(self.game, self.x + self.width // 2, self.y, 'blue')
            self.game.active_events.append(coin)

    def frame_update_remove(self):
        if self.x < self.game.camera_offset_x - self.width:
            return True
        if self.overlap(self.game.player):
            self.handle_collision()
        else:
            if self.game.player.standing_on == self:
                self.game.player.standing_on = None
                self.game.player.gravity_count = self.game.gravity
        return False

    def handle_collision(self):
        s = self.game.GAME_SIZE
        player = self.game.player
        moved_right = player.x - player.x_prev > 0
        moved_left = player.x - player.x_prev < 0
        moved_down = player.y - player.y_prev > 0
        moved_up = player.y - player.y_prev < 0

        # If the player moved down into the wall
        if moved_down and player.y + player.height > self.y and player.y_prev + player.height <= self.y:
            player.y = self.y - player.height
            player.gravity_count = 0
            player.standing_on = self
            player.jumping = False
            
        # If the player moved up into the wall
        if moved_up and player.y < self.y + self.height and player.y_prev >= self.y + self.height:
            player.y = self.y + self.height
            player.gravity_count = self.game.gravity
            player.jumping = False
        # If the player moved right into the wall
        if moved_right and player.x + player.width > self.x and player.x_prev + player.width <= self.x:
            # "Stairs" are when bottom of player and top of wall have small difference
            # If "Stairs" then dont collide. 
            if player.y + player.height < self.y + 0.05 * s:
                player.y = self.y - player.height
            else:
                player.x = self.x - player.width - 0.001
        # If the player moved left into the wall
        if moved_left and player.x < self.x + self.width and player.x_prev >= self.x + self.width:
            # Check stairs
            if player.y + player.height < self.y + 0.05 * s:
                player.y = self.y - player.height
            else:
                player.x = self.x + self.width + 0.001

    def draw(self, window, offset):
        if self.game.player.standing_on == self:
            color = Wall.GREY2
        else:
            color = Wall.GREY
        pygame.draw.rect(window, color, (self.x - offset, self.y, self.width, self.height))

class Bullet(Event):
    COLOR = (255, 0, 0)
    VALUE = 1
    WEIGHT = 5
    
    def __init__(self, game):
        super().__init__(game)
        self.generate()

    def generate(self):
        s = self.game.GAME_SIZE
        self.width = 0.05 * s
        self.height = 0.05 * s
        
        self.x = self.game.player.x + s + random.randint(0, s // 2)
        # 50% chance for y to be between player position
        if random.random() < 0.5:
            self.y = self.game.player.y + random.randint(int(self.game.player.height*0.2), int(self.game.player.height*0.8))
        else:
            self.y = random.randint(int(0.3*s), int(s - self.game.ground_height - 0.2*s))

    def frame_update_remove(self):
        s = self.game.GAME_SIZE
        if self.x + self.width < self.game.camera_offset_x:
            return True
        self.x -= 0.0075 * s  # Move the bullet leftwards
        if self.overlap(self.game.player):
            self.game.game_over = True
            self.game.interactions.append('bullet')
            return True
        return False

    def draw(self, window, offset):
        # Draw a triangle pointing left
        point1 = (self.x - offset, self.y)
        point2 = (self.x - offset + self.width, self.y - self.width // 2)
        point3 = (self.x - offset + self.width, self.y + self.width // 2)
        # Shift all points y by size//2 to match hitbox
        point1 = (point1[0], point1[1] + self.width // 2)
        point2 = (point2[0], point2[1] + self.width // 2)
        point3 = (point3[0], point3[1] + self.width // 2)
        pygame.draw.polygon(window, self.COLOR, [point1, point2, point3])

class Lava(Event):
    COLOR = (255, 125, 0)
    WEIGHT = 5
    
    def __init__(self, game, x=None, width=None):
        super().__init__(game)
        self.generate(x, width)
    
    def generate(self, x=None, width=None):
        s = self.game.GAME_SIZE
        self.height = self.game.ground_height
        self.y = s - self.game.ground_height
        if x and width:
            self.width = width
            self.x = x
        else:
            self.width = random.randint(int(0.05*s), int(0.12*s))
            self.x = self.game.player.x + s + random.randint(0, s // 2)
        
        # Check if lava overlaps with other lava
        for event in self.game.active_events:
            if isinstance(event, Lava) and event.overlap_ratio(self) > 0.1:
                self.fail = True
                return
            
        # Chance for extra lava to the right
        if random.random() < 0.2:
            lava = Lava(self.game, self.x + self.width + self.game.player.width * 2, self.width)
            self.game.active_events.append(lava)

    def frame_update_remove(self):
        if self.x + self.width < self.game.camera_offset_x:
            return True
        if self.overlap(self.game.player):
            self.game.game_over = True
            self.game.interactions.append('lava')
        return False

    def draw(self, window, offset):
        pygame.draw.rect(window, self.COLOR, (self.x - offset, self.y, self.width, self.height))