import numpy as np
import pygame
from bird import Bird
from pipe import Pipe
from priorityReplayBuffer import PriorityReplayBuffer
import random
import matplotlib.pyplot as plt
# constants
# these two are risky to change, may potentially cause issues
SCREEN_DIMENSIONS = (1000, 800)
FPS = 30

ACCELERATION = 10
# epsilon is how often random actions will be taken for exploration, decay is multiplied to the epsilon value once every episode
EPSILON = 1
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.03
# frequency refers to the amount of training steps that need to occur before the target network is updated
TARGET_UPDATE_FREQUENCY = 500

class Flappy:
    # change this to True if you want to play the game yourself.
    playing = False
    rewards = []
    def __init__(self):
        self.width = SCREEN_DIMENSIONS[0]
        self.height = SCREEN_DIMENSIONS[1]
        self.display = pygame.display.set_mode((self.width, self.height))
        self.running = True
        self.generation = 0
        self.epsilon = EPSILON
        self.pipeToDetect = None
        self.experienceReplayBuffer = PriorityReplayBuffer(50000)
        self.step = 0
        self.font = pygame.font.SysFont(None, 32)
        self.ground = pygame.Rect(0, self.height - 10, self.width, 10)
        self.roof = pygame.Rect(0, 0, self.width, 10)
        pygame.display.set_caption("Flappy Learns")
        self.start()
        self.run()

    def displayStats(self):
        """
        Displays reward stats per episode and loss per training step, used for debugging to see plateaus/improvements over time.
        """
        try:
            fig, axs = plt.subplots(2, 1, figsize=(10, 8))
            axs[0].plot(Bird.losses, label='Loss', color='blue')
            axs[0].set_title("Losses")
            axs[0].set_xlabel("Episode")
            axs[0].set_ylabel("Loss")
            axs[0].grid(True)
            axs[0].legend()
            axs[1].plot(Flappy.rewards, label='Reward', color='orange')
            axs[1].set_title("Rewards")
            axs[1].set_xlabel("Episode")
            axs[1].set_ylabel("Reward")
            axs[1].grid(True)
            axs[1].legend()
            plt.tight_layout()
            plt.show()
            plt.close()
        except Exception as x:
            print(x)
            return


    def eventHandler(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if Flappy.playing:
                        self.velocity = self.bird.flap(self.velocity)
                    else:
                        """
                        Displays the stats such as loss and total reward over periods as a graph for debugging
                        """
                        self.displayStats()
            if event.type == pygame.USEREVENT:
                self.spawnPipes()

    def start(self):
        """
        Sets initial conditions for the episode, resets game states and clocks, updates bird intelligence.
        """
        pygame.time.set_timer(event=pygame.USEREVENT, millis=0)
        pygame.time.set_timer(event=pygame.USEREVENT, millis=2300) # controls pipe spawning relative to realtime 
        self.generation += 1
        self.velocity = 0
        self.score = 0
        self.pipes = []
        self.pointFlag = False
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        self.totalReward = 0
        if self.generation == 1:
            self.bird = Bird((self.width, self.height))
        else:
            self.bird = Bird((self.width, self.height), self.bird.intelligence, self.bird.targetIntelligence, experienceBuffer=self.experienceReplayBuffer)
        self.spawnPipes()
        self.pipeToDetect = self.pipes[0]
        if self.generation != 0 and self.generation % 350:
            Bird.learningRate = max(1e-6, Bird.learningRate * 0.5) # decay learning rate
        self.clock = pygame.time.Clock()
        self.display.fill((255,255,255))

    def update(self):
        """
        Deals with the updates to the game on a frame basis, every frame this function is called once.
        Deals with reward handling, appending to experience buffer, and general game loop.
        """
        if self.step % TARGET_UPDATE_FREQUENCY == 0:
            self.bird.setTargetIntelligence() # every n training steps, update the target network
        done = False
        self.reward = 0
        currentState = self.bird.getState(self.pipeToDetect, self.velocity)
        if random.random() > self.epsilon:
            qValues = self.bird.getAction(currentState)
            if np.isnan(qValues).any():
                raise Exception("NANS FOUND IN QVALUE") # shouldnt ever be raised, however if it is, suggests poor parameters used.
            action = np.argmax(qValues)
        else:
            action = random.choice([0,1]) # random action was chosen
        if action == 1 and not Flappy.playing:
            self.velocity = self.bird.flap(self.velocity)
        self.velocity += (ACCELERATION / FPS)
        self.bird.y += self.velocity
        if self.handleCollisions():
            self.reward -= 20
            done = True
        else:
            # proximity reward handling
            self.reward += max(0,1 - currentState[0]) #closer to point line of pipe, bigger reward
            self.reward -= abs(currentState[1]) # the further the agent strays away from the centre of the pipe gap, higher penalty

        for pipe in self.pipes:
            if pipe.x <= 0 - pipe.width:
                del self.pipes[0]
                self.pipes[0].x -= 10 # this is here to prevent pipes from being staggered after deletion

                continue
            pipe.x -= 10
        if self.pipes and not self.pipes[0].scored:
            if self.bird.x >= self.pipes[0].x + self.pipes[0].width: # agent has scored a pipe gap
                self.pipes[0].scored = True
                self.pipes[1].scored = True
                self.pipeToDetect = self.pipes[2]
                self.score += 1
                self.reward += 20
                self.pointFlag = True

        nextState = self.bird.getState(self.pipeToDetect, self.velocity)
        self.experienceReplayBuffer.add((currentState, action, self.reward, nextState, done)) # add to the buffer for training
        self.totalReward += self.reward
        self.bird.learn(self.experienceReplayBuffer)
        self.step += 1
        if done:
            Flappy.rewards.append(self.totalReward)
            self.start()

    def handleCollisions(self):
        """
        Detects collisions between agent and pipes/limits.
        If bird hits a ceiling or floor, it is penalised more heavily.
        """
        birdRect = self.bird.getRect()
        for pipe in self.pipes:
            pipeRect = pipe.getRect()
            if birdRect.colliderect(pipeRect):
                return True
        if birdRect.colliderect(self.ground) or birdRect.colliderect(self.roof):
            self.reward -= 10
            return True
        return False

    def draw(self):
        """
        Handles drawing each frame to the display.
        """
        self.display.fill((180,180,180))
        pygame.draw.rect(self.display, (255,0,0), (self.bird.x, self.bird.y, self.bird.width, self.bird.height))
        for pipe in self.pipes:
           pygame.draw.rect(self.display, (0,210,0), (pipe.x, pipe.y, pipe.width, pipe.height))
        self.drawText()
        pygame.display.flip()
        pygame.display.update()

    def drawText(self):
        """
        Handles the text drawing to the display.
        """
        genText = self.font.render(f"Generation {self.generation} Epsilon:{round(self.epsilon,3)}", True, (0,0,0))
        scoreText = self.font.render(f"{self.score}", True, (0,0,0))
        self.display.blit(genText, (10,10))
        self.display.blit(scoreText, (self.width  // 2, 10))

    def spawnPipes(self):
        """
        Spawns pipes and ensures there is a gap between bottom and top pipe. Rarely causes an error with gap size,
        however seems to be no fix obvious for this issue and doesnt effect learning.
        """
        pipeY = random.randint(0, SCREEN_DIMENSIONS[1]- Pipe.gap)
        self.pipes.append(Pipe((self.width, self.height),pipeY)) # top pipe
        self.pipes.append(Pipe((self.width, self.height), pipeY + Pipe.gap, top=False))

    def run(self):
        """
        Game loop.
        """
        while self.running:
            self.eventHandler()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()


if __name__ == "__main__":
    pygame.font.init()
    flappy = Flappy()

