import math
import random
from abc import ABC, abstractmethod

import pygame

'''
TODO:
- events ideas:
- 1 event is a key, another event is a chest, chest only opens if you have the key
- 3 consequetive coins: 1, 2, 3. only take in that order for reward, wrong order and they dissapear
- 
'''

# TODO:
# if event is outside of screen skip drawing it
class Event(ABC):
    def __init__(self, game):
        self.game = game

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

        # Calculate area of overlap
        overlap_area = overlap_x * overlap_y

        # Calculate areas of the two events
        self_area = self.width * self.height
        event_area = event.width * event.height

        # Use the smaller of the two areas for the ratio
        smaller_area = min(self_area, event_area)
        if smaller_area == 0:
            return 0  # Avoid division by zero

        # Calculate and return the overlap ratio
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
    WEIGHT = 5
    
    def __init__(self, game, x=None, y=None, coin_type=None):
        super().__init__(game)
        self.generate(x, y, coin_type)

    def generate(self, x=None, y=None, coin_type=None):
        s = self.game.size
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

        self.width = self.radius * 2
        self.height = self.radius * 2
        if x and y:
            self.x = x - self.radius
            self.y = y - self.radius * 2
        else:
            self.x = self.game.player.x + s + random.randint(0, s // 2)
            self.y = random.randint(s//2, s - self.game.ground_height - s // 10)
            
        # check if coin overlaps too much with wall
        for event in self.game.active_events:
            if isinstance(event, Wall) and event.overlap_ratio(self) > 0.0:
                return self.generate()

    def frame_update_remove(self):
        if self.overlap(self.game.player):
            self.game.player.score += self.value
            return True
        return False

    def draw(self, window, offset):
        pygame.draw.circle(window, self.color, (self.x - offset + self.radius, self.y + self.radius), self.radius)
        # write the value inside the coin
        '''
        font = pygame.font.SysFont('comicsans', int(self.radius*2))
        text = font.render(str(self.value), 1, (0, 0, 0))
        window.blit(text, (self.x - offset + self.radius - text.get_width() // 2, self.y + self.radius - text.get_height() // 2))
        # if red coin, draw hitbox
        if self.color == Coin.RED:
            pygame.draw.rect(window, (0, 0, 0), (self.x - offset, self.y, self.width, self.height), 1)
        '''

class Wall(Event):
    GREY = (128, 128, 128)
    GREY2 = (28, 28, 28)
    WEIGHT = 15
    
    def __init__(self, game):
        super().__init__(game)
        self.generate()

    def generate(self):
        s = self.game.size
        # Ground wall
        if random.random() < 0.9:
            self.width = random.randint(int(0.2*s), int(0.6*s))
            self.height = random.randint(int(0.1*s), int(0.25*s))
            self.x = self.game.player.x + self.game.size + random.randint(0, self.game.size // 2)
            self.y = self.game.size - self.game.ground_height - self.height
            # check if wall is covered by another wall
            for event in self.game.active_events:
                if isinstance(event, Wall) and event.overlap_ratio(self) > 0.5:
                    return self.generate()
        # Air wall
        else:
            self.width = random.randint(int(0.2*s), int(0.4*s))
            self.height = random.randint(int(0.03*s), int(0.04*s))
            self.x = self.game.player.x + self.game.size + random.randint(0, self.game.size // 2)
            self.y = random.randint(int(0.3*s), int(0.5*s))
            # check if wall is covered by another wall
            for event in self.game.active_events:
                if isinstance(event, Wall) and event.overlap_ratio(self) > 0.1:
                    return self.generate()
            # Also generate a coin on top of it
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
        player = self.game.player
        moved_right = player.x - player.x_prev > 0
        moved_left = player.x - player.x_prev < 0
        moved_down = player.y - player.y_prev > 0
        moved_up = player.y - player.y_prev < 0

        # If the player moved right into the wall
        if moved_right and player.x + player.width > self.x and player.x_prev + player.width <= self.x:
            # if "stairs" then dont collide.
            # stairs are when bottom of player and top of wall have small difference
            if player.y + player.height - self.y < 0.04 * self.game.size:
                player.y = self.y - player.height
            else:
                player.x = self.x - player.width
        # If the player moved left into the wall
        if moved_left and player.x < self.x + self.width and player.x_prev >= self.x + self.width:
            # check stairs
            if player.y + player.height - self.y < 0.04 * self.game.size:
                player.y = self.y - player.height
            else:
                player.x = self.x + self.width
        # If the player moved down into the wall
        if moved_down and player.y + player.height > self.y and player.y_prev + player.height <= self.y:
            player.y = self.y - player.height
            player.gravity_count = 0
            player.standing_on = self
            player.jumping = False
        # If the player moved up into the wall
        if moved_up and player.y < self.y + self.height and player.y_prev >= self.y + self.height:
            player.y = self.y + self.height
            # player.jumping = False

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
        s = self.game.size
        self.width = 0.06 * s
        self.height = 0.06 * s
        
        self.x = self.game.player.x + self.game.size + random.randint(0, self.game.size // 2)
        # 50% chance for y to be between player position
        if random.random() < 0.5:
            self.y = self.game.player.y + random.randint(int(self.game.player.height*0.2), int(self.game.player.height*0.8))
        else:
            self.y = random.randint(int(0.3*s), int(self.game.size - self.game.ground_height - 0.2*s))

    def frame_update_remove(self):
        if self.x + self.width < self.game.camera_offset_x:
            return True
        self.x -= 0.01 * self.game.size  # Move the bullet leftwards
        if self.overlap(self.game.player):
            if self.game.player.star:
                self.game.player.score += self.VALUE
            else:
                self.game.game_over = True
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
        # Also draw a tiny rectangle to show hitbox
        # pygame.draw.rect(window, (0, 0, 0), (self.x - offset, self.y, self.width, self.height), 1)

class Lava(Event):
    COLOR = (255, 125, 0) # Bright red-orange color for lava
    WEIGHT = 5
    # sprite = pygame.image.load("Qrunner/resources/lava_sprite.jpg")
    
    def __init__(self, game, x=None, width=None):
        super().__init__(game)
        self.generate(x, width)
    
    def generate(self, x=None, width=None):
        s = self.game.size
        self.height = self.game.ground_height
        self.y = self.game.size - self.game.ground_height
        if x and width:
            self.width = width
            self.x = x
        else:
            self.width = random.randint(int(0.05*s), int(0.12*s))
            self.x = self.game.player.x + self.game.size + random.randint(0, self.game.size // 2)
        
        # check if lava overlaps with other lava
        for event in self.game.active_events:
            if isinstance(event, Lava) and event.overlap_ratio(self) > 0.1:
                return self.generate()
            
        # x% chance for extra lava to the right
        if random.random() < 0.2:
            lava = Lava(self.game, self.x + self.width + self.game.player.width * 2, self.width)
            self.game.active_events.append(lava)
        
        # self.scaled_sprite = pygame.transform.scale(self.sprite, (self.width, self.height))

    def frame_update_remove(self):
        if self.x + self.width < self.game.camera_offset_x:
            return True
        if self.overlap(self.game.player):
            if not self.game.player.star:
                self.game.game_over = True
        return False

    def draw(self, window, offset):
        #window.blit(self.scaled_sprite, (self.x - offset, self.y))
        pygame.draw.rect(window, self.COLOR, (self.x - offset, self.y, self.width, self.height))
        # Also draw a tiny rectangle to show hitbox
        # pygame.draw.rect(window, (0, 0, 0), (self.x - offset, self.y, self.width, self.height), 1)
        
        '''
        pattern_width = 5  # Width of each stripe in the pattern
        for i in range(self.width // pattern_width):
            # Alternating pattern for visibility in grayscale
            if i % 2 == 0:
                pygame.draw.rect(window, self.COLOR, (self.x - offset + i * pattern_width, self.y, pattern_width, self.height))
            else:
                pygame.draw.rect(window, (200, 200, 200), (self.x - offset + i * pattern_width, self.y, pattern_width, self.height))
        '''

class Star(Event):
    YELLOW = (255, 255, 0)  # Bright yellow color for the star
    SIZE = 25
    WEIGHT = 1

    def __init__(self, game):
        super().__init__(game)
        self.generate()

    def generate(self):
        self.frames_left = 60*5
        self.taken = False
        self.width = self.SIZE * 2
        self.height = self.SIZE * 2
        self.x = self.game.player.x + self.game.size + random.randint(0, self.game.size // 2)
        self.y = random.randint(100, self.game.size - self.game.ground_height - 50)

    def frame_update_remove(self):
        # Remove star if it is used up
        if self.frames_left <= 0:
            return True
        # Remove star if it is outside of screen and never activated
        if (self.x + self.SIZE < self.game.camera_offset_x) and not self.taken:
            return True
        # Remove star if it was taken by the player but not the players current star
        if self.game.player.star != self and self.taken:
            return True
    
        if self.overlap(self.game.player) and not self.taken:
            self.game.player.star = self
            self.taken = True
        if self.taken:
            self.frames_left -= 1
            if self.frames_left <= 0:
                self.game.player.star = None
                    
        return False

    def draw_star(self, center, size, points, color, window):
        """Helper function to draw a star"""
        def get_star_points(center, size, points):
            """Calculate the (x, y) coordinates for a star"""
            step = math.pi / points
            angle = 0
            outer_points = []
            inner_points = []
            for _ in range(points):
                outer_x = center[0] + int(math.cos(angle) * size)
                outer_y = center[1] + int(math.sin(angle) * size)
                outer_points.append((outer_x, outer_y))

                inner_angle = angle + step / 2
                inner_x = center[0] + int(math.cos(inner_angle) * size / 2)
                inner_y = center[1] + int(math.sin(inner_angle) * size / 2)
                inner_points.append((inner_x, inner_y))

                angle += 2 * step

            return [val for pair in zip(outer_points, inner_points) for val in pair]

        points = get_star_points(center, size, points)
        pygame.draw.polygon(window, color, points)

    def draw(self, window, offset):
        # Draw a star
        if not self.taken:
            center = (self.x - offset + self.SIZE, self.y + self.SIZE)
            self.draw_star(center, self.SIZE, 5, self.YELLOW, window)

class Portal(Event): # Probably make imune for 50 frames something
    ENTRY_COLOR = (128, 0, 128)  # Dark purple color for the entry portal
    EXIT_COLOR = (204, 153, 255)  # Light purple color for the exit portal
    WIDTH = 20
    HEIGHT = 60
    WEIGHT = 1
    
    def __init__(self, game):
        super().__init__(game)
        self.generate()

    def generate(self):
        self.width = self.WIDTH
        self.height = self.HEIGHT
        self.x, self.y = self.generate_portal_position()
        self.exit_x, self.exit_y = self.generate_portal_position()
        # Ensuring exit portal is always to the right of the entry portal
        if self.exit_x < self.x:
            self.exit_x, self.x = self.x, self.exit_x
        # If the distance is less than x, then add x to the exit portal's x position
        if self.exit_x - self.x < 200:
            self.exit_x += 200

    def generate_portal_position(self):
        x = self.game.player.x + self.game.size + random.randint(0, self.game.size // 4)
        y = random.randint(100, self.game.size - self.game.ground_height - 100)
        return x, y

    def frame_update_remove(self):
        if self.x < self.game.camera_offset_x - self.WIDTH and self.exit_x < self.game.camera_offset_x - self.WIDTH:
            return True
        if self.overlap(self.game.player):
            # Teleport the player to the exit portal
            self.game.player.x = self.exit_x
            self.game.player.y = self.exit_y
            # Not jumping anymore and reset gravity
            self.game.player.jumping = False
            self.game.player.gravity_count = self.game.gravity
            return True
        return False

    def draw(self, window, offset):
        # Draw entry portal
        pygame.draw.rect(window, self.ENTRY_COLOR, (self.x - offset, self.y, self.WIDTH, self.HEIGHT))
        # Draw exit portal
        pygame.draw.rect(window, self.EXIT_COLOR, (self.exit_x - offset, self.exit_y, self.WIDTH, self.HEIGHT))