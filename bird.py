import numpy as np
import pygame
from DQN.dense import DenseLayer
from DQN.relu import Relu
from DQN.loss import meanSquaredError
import copy
SIZE = (50,30)

class Bird:
    def __init__(self, screenSize):
        self.width = SIZE[0]
        self.height = SIZE[1]
        self.x, self.y = (screenSize[0] // 6, screenSize[1] // 2)
        self.intelligence = [
            DenseLayer(3, 16),
            Relu(),
            DenseLayer(16,24),
            Relu(),
            DenseLayer(24,2)
        ]
        self.setTargetIntelligence()

    def flap(self, velocity):
        if velocity > 0:
            return -7
        elif velocity > -14:
            return velocity - 3
        return velocity
    
    def setTargetIntelligence(self):
        self.targetIntelligence = copy.deepcopy(self.intelligence)
    
    def getState(self, pipe, velocity):
        x = pipe.x + pipe.width
        y = pipe.y + (pipe.height) + 75
        dx, dy = abs(self.x - x) / 1000, abs(self.y - y)/800
        velocity = velocity / 15
        return np.array([dx, dy, velocity])
    
    def getAction(self):
        pass



    def getRect(self):
         return pygame.Rect(self.x, self.y, self.width, self.height)

