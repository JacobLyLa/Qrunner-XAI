import pygame

# Keep track of the player states, also used for rendering the player
class Player:
    BLUE = (0, 0, 255)
    def __init__(self, x, y, width, height, velocity_x, velocity_y):
        self.x = x
        self.y = y
        self.x_prev = x
        self.y_prev = y
        self.width = width
        self.height = height

        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.score = 0
        self.score_prev = 0
        self.frames_since_reward = 0
        self.gravity_count = 0
        self.jumping = False
        self.dodging = False
        self.standing_on = None

    def draw(self, screen, offset):
        pygame.draw.rect(screen, Player.BLUE, (self.x - offset, self.y, self.width, self.height))