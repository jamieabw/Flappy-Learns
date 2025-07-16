import numpy as np
SIZE = (50,30)

class Bird:
    def __init__(self, screenSize):
        self.width = SIZE[0]
        self.height = SIZE[1]
        self.x, self.y = (screenSize[0] // 6, screenSize[1] // 2)

    def flap(self, velocity):
        if velocity > 0:
            return -7
        elif velocity > -14:
            return velocity - 3
        return velocity


    def getRect():
        pass

