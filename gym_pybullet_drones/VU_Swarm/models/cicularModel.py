from distutils.log import error
import numpy as np


class CircularAgent(object):
    def __init__(self, radius, kPosition, center_circle, env):
        self.radius = radius
        self.kPosition = kPosition
        self.center_circle = center_circle
        self.env = env
        self.waypoints = [[1, 1], [1, 0], [0, 0], [0, 1]]
        self.current_wp = 0

    def setParams(self, total_time, startTime):
        self.totalTime = total_time
        self.startTime = startTime
        self.current_wp = np.random.choice([0, 1, 2, 3])

    def choose_action(self, observation):
        """
        Observation: List with state (X,Y)
        """
        errorX = self.waypoints[self.current_wp] - observation[len(observation) - 2 :]
        # print(np.sum(np.abs(errorX)))
        if np.sum(np.abs(errorX)) < 0.25:
            # print('Changed wp')
            if self.current_wp == 3:
                self.current_wp = 0
            else:
                self.current_wp += 1
        vx = errorX[0] * self.kPosition
        vy = errorX[1] * self.kPosition
        return np.array([vx, vy])

    def learn(self):
        # print("I'm the circular")
        pass

    def remember(self, *args):
        pass
