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
        self.generation = 0
        self.pipeToDetect = None
        self.font = pygame.font.SysFont(None, 64)
        self.ground = pygame.Rect(0, self.height - 10, self.width, 10)
        self.roof = pygame.Rect(0, 0, self.width, 10)
        pygame.display.set_caption("Flappy Learns")
        self.start()
        self.run()

    def eventHandler(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.velocity = self.bird.flap(self.velocity)
            if event.type == pygame.USEREVENT:
                self.spawnPipes()

    def start(self):
        pygame.time.set_timer(event=pygame.USEREVENT, millis=0)
        pygame.time.set_timer(event=pygame.USEREVENT, millis=2300)
        self.generation += 1
        self.velocity = 0
        self.score = 0
        self.pipes = []
        self.pointFlag = False
        self.bird = Bird((self.width, self.height))
        self.spawnPipes()
        self.pipeToDetect = self.pipes[0]
        self.clock = pygame.time.Clock()
        self.display.fill((255,255,255))

    def update(self):
        self.velocity += (ACCELERATION / 60)
        self.bird.y += self.velocity
            #print(2)
        print(self.bird.getState(self.pipeToDetect, self.velocity))
        
        for pipe in self.pipes:
            if pipe.x <= 0 - pipe.width:
                del self.pipes[0]
                self.pipes[0].x -= 10 

                continue
                #print(self.pipes)
                # need to remove the two pipes here
            pipe.x -= 10
        self.handleCollisions()
        if self.pipes and not self.pipes[0].scored:
            if self.bird.x >= self.pipes[0].x + self.pipes[0].width:
                self.pipes[0].scored = True
                self.pipes[1].scored = True
                self.pipeToDetect = self.pipes[2]
                self.score += 1
                self.pointFlag = True
                print(self.score)

    def handleCollisions(self):
        birdRect = self.bird.getRect()
        for pipe in self.pipes:
            pipeRect = pipe.getRect()
            if birdRect.colliderect(pipeRect):
                self.start()
        if birdRect.colliderect(self.ground) or birdRect.colliderect(self.roof):
            self.start()

    def draw(self):
        self.display.fill((220,220,220))
        pygame.draw.rect(self.display, (255,0,0), (self.bird.x, self.bird.y, self.bird.width, self.bird.height))
        for pipe in self.pipes:
           pygame.draw.rect(self.display, (0,210,0), (pipe.x, pipe.y, pipe.width, pipe.height))
        self.drawText()
        pygame.display.flip()
        pygame.display.update()

    def drawText(self):
        genText = self.font.render(f"Generation {self.generation}", True, (0,0,0))
        scoreText = self.font.render(f"{self.score}", True, (0,0,0))
        self.display.blit(genText, (10,10))
        self.display.blit(scoreText, (self.width  // 2, 10))

    def spawnPipes(self):
        pipeY = random.randint(150, SCREEN_DIMENSIONS[1]- 150)
        self.pipes.append(Pipe((self.width, self.height),pipeY)) # top pipe
        self.pipes.append(Pipe((self.width, self.height), pipeY + 150, top=False))
        #print("spawned")

    def run(self):
        # pygame.time.set_timer(event=pygame.USEREVENT, millis=2300)
        while self.running:
            self.eventHandler()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()





if __name__ == "__main__":
    pygame.font.init()
    flappy = Flappy()

