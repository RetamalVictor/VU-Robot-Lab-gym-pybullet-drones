import time
import numpy as np
from pprint import pprint

from gym_pybullet_drones.utils.utils import sync

import matplotlib.pyplot as plt
from datetime import datetime

"""
- Create a target agent to follow
- Generate a reward based on the distance from the agetn
- Initialize the learning agents with the same controller
- Initialize a target network
---- Training
- Init the environment in the reset position
- After the collection of the trajectory (for every agent)
- Update the target network when it's time

"""


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

    - The observation does not include the neigbors. So they just avoid the target to not collide.
    """

    def __init__(self, env, agent, target_agent, args, checkpoint=False, logger=None):
        """
        TO-DO:
        [ ] Describe the parameters here
        """
        if logger is not None:
            self.logger = logger
            self.logging = True
        else:
            self.logging = False
        self.checkpoint = checkpoint
        self.env = env
        self.dt = 1 / env.SIM_FREQ
        self.target = args["target"]
        self.gui = args["gui"]
        self.idle_time = args["idle_time"]
        self.cntrl_freq = args["cntrl_freq"]
        self.aggr_phy_steps = args["aggr_phy_steps"]
        self.num_drones = args["num_drones"]
        self.training_drones = self.num_drones - 1
        self.debugHeadigs = args["debug"]
        self.agent = [agent for _ in range(self.training_drones)]
        self.agent.append(target_agent)
        self.action = {str(i): np.array([0, 0, 0, 1]) for i in range(self.num_drones)}

        if self.checkpoint:
            for agent in range(self.training_drones):
                self.agent[agent].load_models()

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
        score_history_1 = []
        score_prev = -np.inf
        for i in range(episodes):
            score = self.trainingLoop(duration_sec=10)
            score_history_1.append(score)
            print(f"Episode: {i}")
            pprint(score)
            print()
            if score[str(0)] > score_prev:
                self.agent[0].save_models()
                score_prev = score[str(0)]
            # print(
            #     f"episode ",
            #     i,
            #     "score %.2f" % score,
            #     "trailing 10 games avg %.3f" % np.mean(score_history_1[-10:]),
            # )
        filename = f"drones-alpha{self.alpha}-beta{self.beta}-{datetime.now()}.png"
        plotLearning(score_history_1, filename, window=100)

    def trainingLoop(self, duration_sec=None):
        """
        WORKING HERE:
        Implementing the linear algebra solution for rewards
        """
        obs = self.reset()
        score = {str(i): 0 for i in range(self.training_drones)}
        done = False
        START = time.time()
        self.agent[-1].setParams(
            int(duration_sec * self.env.SIM_FREQ) // self.aggr_phy_steps, START
        )

        action = [
            self.agent[i].choose_action(
                np.array([obs[i], obs[-1]]).reshape(
                    4,
                )
            )
            for i in range(len(self.agent))
        ]
        action_dict = {
            str(i): np.append(action[i], [0, 1]) for i in range(len(self.agent))
        }
        START = time.time()
        self.agent[-1].setParams(int(duration_sec * self.env.SIM_FREQ), START)
        for i in range(0, int(duration_sec * self.env.SIM_FREQ), self.aggr_phy_steps):
            if done:
                break
            next_obs, rewards, done, _ = self.env.step(action_dict)
            # print(np.array([obs[0],obs[-1]]).reshape(4,), action[0], rewards[0], np.array([next_obs[0], next_obs[-1]]).reshape(4,), int(done))
            # break
            for agent in range(len(self.agent) - 1):
                self.agent[agent].remember(
                    np.array([obs[agent], obs[-1]]).reshape(
                        4,
                    ),
                    action[agent],
                    rewards[agent],
                    np.array([next_obs[agent], next_obs[-1]]).reshape(
                        4,
                    ),
                    int(done),
                )
            for agent in self.agent:
                agent.learn()
            for agent in range(self.training_drones):
                score[str(agent)] += rewards[agent]
            obs = next_obs

            if self.debugHeadigs:
                for j in range(self.num_drones):
                    self.env._debugHeadings(j, self.headings)

            if i % self.cntrl_freq == 0:
                # action = [self.agent[i].choose_action(obs[i]) for i in range(len(self.agent))]
                action = [
                    self.agent[i].choose_action(
                        np.array([obs[i], obs[-1]]).reshape(
                            4,
                        )
                    )
                    for i in range(len(self.agent))
                ]
                action_dict = {
                    str(i): np.append(action[i], [0, 1]) for i in range(len(self.agent))
                }

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
