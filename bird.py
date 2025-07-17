import numpy as np
import pygame
from DQN.dense import DenseLayer
from DQN.relu import Relu
from DQN.loss import meanSquaredError, dMeanSquaredError
import random
import copy
SIZE = (50,30)
BATCH_SIZE = 16
GAMMA = 0.5
class Bird:
    def __init__(self, screenSize, intelligence=None, targetIntelligence=None, experienceBuffer=None):
        self.width = SIZE[0]
        self.height = SIZE[1]
        self.x, self.y = (screenSize[0] // 6, screenSize[1] // 2)
        if intelligence is None:
            self.intelligence = [
                DenseLayer(3, 16),
                Relu(),
                DenseLayer(16,24),
                Relu(),
                DenseLayer(24,2)
            ]
            self.setTargetIntelligence()
        else:
            # place backpropagation here for learning
            self.intelligence = intelligence
            if targetIntelligence is None:
                self.setTargetIntelligence()
            else:
                self.targetIntelligence = targetIntelligence
            self.learn(experienceBuffer)


    def flap(self, velocity):
        if velocity > 0:
            return -7
        elif velocity > -14:
            return velocity - 3
        return velocity
    
    def learn(self, buffer):
        batch = random.sample(buffer, BATCH_SIZE)
        losses = []
        for experience in batch:
            state, action, reward, nextState, done = experience
            qValues = self.getAction(state)
            nextQValue = np.max(self.getAction(nextState, target=True))
            if done:
                y = reward
            else:
                y = reward + (GAMMA * nextQValue)
            targetQValues = np.copy(qValues)
            targetQValues[action] = y
            loss = meanSquaredError(qValues, targetQValues)
            losses.append(loss)
            error = dMeanSquaredError(qValues, targetQValues)
            output = error
            for layer in reversed(self.intelligence):
                output = layer.backPropagation(output)
        print(np.mean(losses))
    
    def setTargetIntelligence(self):
        self.targetIntelligence = copy.deepcopy(self.intelligence)
    
    def getState(self, pipe, velocity):
        x = pipe.x + pipe.width
        y = pipe.y + (pipe.height) + 75
        dx, dy = abs(self.x - x) / 1000, abs(self.y - y)/800
        velocity = velocity / 15
        return np.array([dx, dy, velocity])
    
    def getAction(self, state, target=False):
        if not target:
            input = state
            for layer in self.intelligence:
                #print(layer)
                input = layer.forwardPropagation(input)
            return input
        else:
            input = state
            for layer in self.targetIntelligence:
                #print(layer)
                input = layer.forwardPropagation(input)
            return input



    def getRect(self):
         return pygame.Rect(self.x, self.y, self.width, self.height)

