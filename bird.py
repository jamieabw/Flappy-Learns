import numpy as np
import pygame
from DQN.dense import DenseLayer
from DQN.relu import Relu
from DQN.leakyRelu import LeakyRelu
from DQN.loss import meanSquaredError, dMeanSquaredError
import random
import copy
SIZE = (50,30)
BATCH_SIZE = 64
GAMMA = 0.99
class Bird:
    losses = []
    learningRate = 3e-4
    def __init__(self, screenSize, intelligence=None, targetIntelligence=None, experienceBuffer=None):
        self.width = SIZE[0]
        self.height = SIZE[1]
        self.x, self.y = (screenSize[0] // 6, screenSize[1] // 2)
        if intelligence is None:
            self.intelligence = [
                DenseLayer(4, 24),
                LeakyRelu(),
                DenseLayer(24,18),
                LeakyRelu(),
                DenseLayer(18,2)

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
        return -9
    
    def learn(self, buffer, double=True):
        if len(buffer) < 1000:
            #print(len(buffer))
            return
        batch = random.sample(buffer, BATCH_SIZE)
        losses = []
        for experience in batch:
            state, action, reward, nextState, done = experience
            qValues = self.getAction(state)
            if double:
                bestAction = np.argmax(self.getAction(nextState))
                targetQ = self.getAction(nextState, target=True)
                nextQValue = targetQ[bestAction]
            else:
                nextQValue = np.max(self.getAction(nextState, target=True))
            if done:
                y = reward
            else:
                y = reward + (GAMMA * nextQValue)
            nextQValue = np.clip(nextQValue, -1.0, 1.0)
            targetQValues = np.copy(qValues)
            targetQValues[action] = y
            #print(targetQValues)
            loss = meanSquaredError(qValues, targetQValues)
            losses.append(loss)
            error = dMeanSquaredError(qValues, targetQValues)
            output = error
            #print(output)
            for layer in reversed(self.intelligence):
                output = layer.backPropagation(np.nan_to_num(output, nan=0.0), learningRate=Bird.learningRate)
        Bird.losses.append(np.mean(losses))
    
    def setTargetIntelligence(self):
        print("Updated target network")
        self.targetIntelligence = copy.deepcopy(self.intelligence)
    
    def getState(self, pipe, velocity):
        x = pipe.x + pipe.width
        y = pipe.y + (pipe.height) + 125
        dx, dy = abs(self.x - x) / 1000, abs(self.y - y)/800
        yPos = self.y / 800
        velocity = velocity / 15
        return np.array([dx, dy, velocity, yPos])
    
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
                input = layer.forwardPropagation(np.nan_to_num(input, nan=0.0))
            return input



    def getRect(self):
         return pygame.Rect(self.x, self.y, self.width, self.height)

