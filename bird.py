import numpy as np
import pygame
from DQN.dense import DenseLayer
from DQN.relu import Relu
from DQN.leakyRelu import LeakyRelu
from DQN.loss import meanSquaredError, dMeanSquaredError, huberLoss, dHuberLoss
import copy
SIZE = (50,30)
# these constants will affect the model's learning significantly, change with caution. GAMMA should never be below 0.95.
BATCH_SIZE = 64
GAMMA = 0.98
LOSS = meanSquaredError
LOSS_DERIVATIVE = dMeanSquaredError
LEARNING_RATE = 0.001
class Bird:
    losses = []
    learningRate = LEARNING_RATE
    def __init__(self, screenSize, intelligence=None, targetIntelligence=None, experienceBuffer=None):
        self.width = SIZE[0]
        self.height = SIZE[1]
        self.x, self.y = (screenSize[0] // 6, screenSize[1] // 2)
        if intelligence is None:
            # current architecture is 4,24,2 this can be changed for better complexity.
            self.intelligence = [
                DenseLayer(4, 24),
                LeakyRelu(),
                DenseLayer(24,2),
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
        """
        Flaps, increases birds velocity to 9
        """
        return -9
    
    def learn(self, buffer, double=True):
        """
        Handles the DQL process, uses double DQL if double is true, applies back propagation.
        """
        if len(buffer.buffer) < 1000:
            return # not enough experiences yet
        losses = []
        tdErrors = []
        batch, indices, weights = buffer.sample(BATCH_SIZE)
        for index, experience in enumerate(batch):
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
            targetQValues = np.copy(qValues)
            targetQValues[action] = y
            tdErrors.append(y - qValues[action])
            loss = LOSS(qValues, targetQValues, action)
            losses.append(loss)
            error = LOSS_DERIVATIVE(qValues, targetQValues, action)
            output = error
            for layer in reversed(self.intelligence):
                output = layer.backPropagation(np.nan_to_num(output, nan=0.0), learningRate=Bird.learningRate)
        buffer.updatePriorities(indices, tdErrors)
        Bird.losses.append(np.mean(losses))
    
    def setTargetIntelligence(self):
        print("Updated target network")
        self.targetIntelligence = copy.deepcopy(self.intelligence)
    
    def getState(self, pipe, velocity) -> np.array:
        """
        Gets the 'state' (what will be used as input to the DQN)
        """
        x = pipe.x + pipe.width
        y = pipe.y + (pipe.height) + 125
        dx, dy = abs(self.x - x) / 1000, abs(self.y - y)/800
        yPos = self.y / 800
        velocity = velocity / 15
        return np.array([dx, dy, velocity, yPos])
    
    def getAction(self, state, target=False) -> np.array:
        """
        Handles forward propagation of the main and target DQNs.
        """
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

