import numpy as np
import pygame
from bird import Bird
from pipe import Pipe
import random
SCREEN_DIMENSIONS = (1000, 800)
FPS = 30
ACCELERATION = 10 * 2

class Flappy:
    def __init__(self):
        self.width = SCREEN_DIMENSIONS[0]
        self.height = SCREEN_DIMENSIONS[1]
        self.display = pygame.display.set_mode((self.width, self.height))
        self.running = True
        self.velocity = 0
        self.pipes = []
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("Flappy Learns")
        self.bird = Bird((self.width, self.height))
        self.run()

    def eventHandler(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.velocity = self.bird.flap(self.velocity)

    def update(self):
        self.velocity += (ACCELERATION / 60)
        self.bird.y += self.velocity
        if pygame.time.get_ticks() % 2000 < 50: # change this to actually be 2 secs
            pipeY = random.randint(150, SCREEN_DIMENSIONS[1]- 150)
            self.pipes.append(Pipe((self.width, self.height),pipeY)) # top pipe
            self.pipes.append(Pipe((self.width, self.height), pipeY + 150, top=False))
            print(2)
        for pipe in self.pipes:
            pipe.x -= 10
            if pipe.x <= 0 - pipe.width:
                pass
                #print(self.pipes)
                # need to remove the two pipes here

    def draw(self):
        self.display.fill((255,255,255))
        pygame.draw.rect(self.display, (255,0,0), (self.bird.x, self.bird.y, self.bird.width, self.bird.height))
        for pipe in self.pipes:
           pygame.draw.rect(self.display, (0,0,0), (pipe.x, pipe.y, pipe.width, pipe.height)) 
        pygame.display.flip()
        pygame.display.update()

    def run(self):
        while self.running:
            self.eventHandler()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()





if __name__ == "__main__":
    flappy = Flappy()

