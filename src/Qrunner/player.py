import pygame

# Keep track of the player states, also used for rendering the player
class Player:
    BLUE = (0, 0, 255)
    ORANGE = (255, 165, 0)
    def __init__(self, x, y, width, height, velocity_x, velocity_y):
        self.x = x
        self.y = y
        self.x_prev = x
        self.y_prev = y
        self.width = width
        self.height = height

        self.coins_picked = []
        self.coins_missed = []
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.score = 0
        self.score_prev = 0
        self.frames_since_reward = 0
        self.gravity_count = 0
        self.jumping = False
        self.dodging = False
        self.standing_on = None
        self.star = None
        
        # Change manually if wanted when debugging
        self.draw_coords = False
        self.draw_score = False
        self.draw_star = False
        
        self.font = pygame.font.Font('freesansbold.ttf', 5)

    def draw(self, screen, offset):
        player_color = Player.ORANGE if self.star else Player.BLUE
        pygame.draw.rect(screen, player_color, (self.x - offset, self.y, self.width, self.height))

        if self.draw_coords:
            text = self.font.render(f'x: {int(self.x)}, y: {int(self.y)}', True, (0, 0, 0))
            textRect = text.get_rect()
            textRect.topleft = (0, 0)
            screen.blit(text, textRect)

        if self.draw_score:
            text = self.font.render(f'Score: {self.score:.2f}', True, (0, 0, 0))
            textRect = text.get_rect()
            textRect.topleft = (0, 10)
            screen.blit(text, textRect)

        if self.draw_star:
            if self.star:
                star_text = f'Star: {int(self.star.active_frames//60)}'
            else:
                star_text = 'Star: '
            text = self.font.render(star_text, True, (0, 0, 0))
            textRect = text.get_rect()
            textRect.topleft = (0, 20)
            screen.blit(text, textRect)
