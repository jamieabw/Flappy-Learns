import pygame
# will implement randomness by choosing random y point, using a fixed displacement to get the other point 
DEFAULT_PIPE_WIDTH = 140
class Pipe:
    width = DEFAULT_PIPE_WIDTH
    def __init__(self, screenSize, y, top=True):
        self.x = screenSize[0] + 200
        self.width = DEFAULT_PIPE_WIDTH
        self.scored = False
        if top:
            self.y = 0#screenSize[1]
            self.height = y
        else:
            self.y = y
            self.height = screenSize[1] - y

    def getRect(self):
         return pygame.Rect(self.x, self.y, self.width, self.height)

         
