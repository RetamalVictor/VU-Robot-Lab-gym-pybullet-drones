import time
import numpy as np


from gym_pybullet_drones.utils.utils import sync

import matplotlib.pyplot as plt
from datetime import datetime


def plotLearning(scores, filename, x=None, window=5):
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - window) : (t + 1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel("Score")
    plt.xlabel("Game")
    plt.plot(x, running_avg)
    plt.savefig(filename)


class NeuralControl:
    """
    TO-DO:
    - extract take off and landing to a parent class
    - Learning method. Include the trajectory colletion inside this class
    """

    def __init__(self, env, agent, args, logger=None):
        """
        TO-DO:
        [ ] Describe the parameters here
        """
        if logger is not None:
            self.logger = logger
            self.logging = True
        else:
            self.logging = False
        self.env = env
        self.dt = 1 / env.SIM_FREQ
        self.target = args["target"]
        self.gui = args["gui"]
        self.idle_time = args["idle_time"]
        self.cntrl_freq = args["cntrl_freq"]
        self.aggr_phy_steps = args["aggr_phy_steps"]
        self.num_drones = args["num_drones"]
        self.debugHeadigs = args["debug"]
        self.agent = agent

    def plot(self):
        self.logger.plot()

    def init(self):
        START = time.time()
        for i in range(1, int(self.idle_time * self.env.SIM_FREQ), self.aggr_phy_steps):

            _, _, _, _ = self.env.step(self.action)

            if i % self.env.SIM_FREQ == 0:
                self.env.render()

            if self.gui:
                sync(i, START, self.env.TIMESTEP)

    def takeOff(self, target_height, duration):
        self.action = {str(i): np.array([0, 0, 1, 1]) for i in range(self.num_drones)}
        START = time.time()
        for i in range(1, int(duration * self.env.SIM_FREQ), self.aggr_phy_steps):

            _, _, _, _ = self.env.step(self.action)
            if self.env._getDroneStateVector(0)[2] == target_height:
                break

            if i % self.cntrl_freq == 0:
                for j in range(self.num_drones):
                    self.action[str(j)] = np.array([0, 0, 1, 1 / i])

            if i % self.env.SIM_FREQ == 0:
                self.env.render()

            if self.gui:
                sync(i, START, self.env.TIMESTEP)

    def train(self, episodes):
        score_history = []
        for i in range(episodes):
            score = self.trainingLoop(duration_sec=10)
            score_history.append(score)
            print(
                "episode ",
                i,
                "score %.2f" % score,
                "trailing 10 games avg %.3f" % np.mean(score_history[-10:]),
            )

        filename = f"drones-alpha{self.alpha}-beta{self.beta}-{datetime.now()}.png"
        plotLearning(score_history, filename, window=100)

    def trainingLoop(self, duration_sec=None):
        obs = self.reset()
        score = 0
        done = False
        action = self.agent.choose_action(obs)
        action_dict = {str(0): np.append(action, [0, 1])}
        START = time.time()
        for i in range(0, int(duration_sec * self.env.SIM_FREQ), self.aggr_phy_steps):
            if done:
                break
            next_obs, rewards, done, _ = self.env.step(action_dict)
            self.agent.remember(obs, action, rewards, next_obs, int(done))
            self.agent.learn()
            score += rewards
            obs = next_obs

            if self.debugHeadigs:
                for j in range(self.num_drones):
                    self.env._debugHeadings(j, self.headings)

            if i % self.cntrl_freq == 0:
                action = self.agent.choose_action(obs)
                action_dict = {str(0): np.append(action, [0, 1])}

            if self.gui:
                sync(i, START, self.env.TIMESTEP)
        return score

    def testAgent(self, duration):
        self.env.GUI = True
        obs = self.env.reset()
        action = self.agent.choose_action(obs)
        action_dict = {str(0): np.append(action, [0, 1])}
        START = time.time()
        for i in range(1, int(duration * self.env.SIM_FREQ), self.aggr_phy_steps):

            obs, rewards, done, _ = self.env.step(action_dict)

            if i % self.cntrl_freq == 0:
                action = self.agent.choose_action(obs)
                action_dict = {str(0): np.append(action, [0, 1])}

            # if self.logging:
            #     self._logSimulation(i, obs, 2)

            if i % self.env.SIM_FREQ == 0:
                self.env.render()

            if self.gui:
                sync(i, START, self.env.TIMESTEP)

    def close(self):
        self.env.close()

    def reset(self):
        obs = self.env.reset()
        return obs

    def _logSimulation(self, tick, obs, drones=None):  # Under Construction
        if drones is None:
            drones = self.num_drones
        for j in range(drones):
            self.logger.log(
                drone=j,
                timestamp=tick / self.env.SIM_FREQ,
                # state=obs[str(j)]["state"],
                state=obs,
            )
