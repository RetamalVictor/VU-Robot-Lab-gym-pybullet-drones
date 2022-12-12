import time
import numpy as np
import os
from pprint import pprint
import copy
from gym_pybullet_drones.utils.utils import sync

import matplotlib.pyplot as plt
from datetime import datetime
from gym_pybullet_drones.VU_Swarm.utils import plotLearning
from torch.utils.tensorboard import SummaryWriter

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


class NeuralControl:
    """
    TO-DO:
    - extract take off and landing to a parent class
    - Learning method. Include the trajectory colletion inside this class
    - Plotting not only one reward, but also average.

    - The observation does not include the neigbors. So they just avoid the target to not collide.
    """

    def __init__(self, env, agent, args, checkpoint=False, mode="Eval", logger=None):
        """
        TO-DO:
        [ ] Describe the parameters here
        """
        assert mode in ["Train", "Eval"]
        if logger is not None:
            self.logger = logger
            self.logging = True
        else:
            self.logging = False
        self.checkpoint = checkpoint
        self.env = env
        self.dt = 1 / env.SIM_FREQ
        # self.chkpt_dir = args["checkpoint_dir"]
        self.target = args["target"]
        self.gui = args["gui"]
        self.idle_time = args["idle_time"]
        self.cntrl_freq = args["cntrl_freq"]
        self.aggr_phy_steps = args["aggr_phy_steps"]
        self.num_drones = args["num_drones"]
        self.num_close_drones = args["NNeigh"]
        self.training_drones = self.num_drones
        self.debugHeadigs = args["debug"]
        self.heading = args["heading"]
        exp_name = args["exp_name"]
        self.logging_dir = os.path.join(
            args["logging_dir"], os.path.join("tensorboard", f"Experiment_{exp_name}l")
        )
        if not os.path.exists(self.logging_dir):
            os.makedirs(self.logging_dir)
        # self.agent = [copy.copy(agent) for _ in range(self.training_drones)]
        self.agent = agent
        if self.heading == True:
            for agent in self.agent:
                agent.initHeading()
        self.action = {str(i): np.array([0, 0, 0, 1]) for i in range(self.num_drones)}
        self.mode = mode
        if self.checkpoint:
            for agent in range(self.training_drones):
                self.agent[agent].load_models()

    ################################################################################

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

    ################################################################################

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

    ################################################################################

    def train(self, episodes, duration_sec=10):
        """
        Main training method
        """
        score = {str(i): np.zeros((episodes)) for i in range(self.training_drones)}
        score_prev = -np.inf
        self.writer = SummaryWriter(self.logging_dir)

        # Main training loop (Now, cap to num of episodes)
        for i in range(episodes):
            # Collect trajectory
            print(f"Episode: {i}")
            START = time.time()
            episode_score, A_loss, C_loss = self.trainingLoop(duration_sec)
            print(f"Episode {i} duration: {time.time() - START}")
            # for agent in range(self.training_drones): score_list[agent].append(score[str(agent)])
            # for agent in range(self.training_drones): score[str(agent)][i] = episode_score[i]
            self.writer.add_scalar("Agent 1 reward", episode_score[0], i)
            self.writer.add_scalar("Agent 2 reward", episode_score[1], i)
            self.writer.add_scalar("Actor loss", A_loss, i)
            self.writer.add_scalar("Critic loss", C_loss, i)
            self.writer.add_scalar("Mean reward", np.mean(episode_score), i)
            # Information about training
            if i % 5 == 0:
                pprint(episode_score)
                print()

            # Saving best models
            if self.mode == "Train":
                if np.mean(episode_score) > score_prev and i > 10:
                    self.agent[0].save_models()
                    score_prev = np.mean(episode_score)

                # # Ploting results
                # filename = f"drones-alpha-beta-flocking-tasks.png"
                # plotLearning(score_list, filename, window=10)
        self.writer.close()

    ################################################################################

    def _preprocesObservationTraining(self, obs, nth_drone):
        """
        This is deprecated with the addition of _preprocessObs in the env class
        """
        return obs[nth_drone]

    ################################################################################

    def trainingLoop(self, duration_sec=None):
        """
        Method to collect the trajectory of an episode
        """
        # Initializing environment
        obs = self.reset()
        # score = {str(i): 0 for i in range(self.training_drones)}
        score = np.zeros((self.training_drones, 1))
        A_loss = 0
        C_loss = 0
        done = False
        START = time.time()

        # Setting params for target trajectory for action initialization ( this can be changed for something else later )
        # CHECK IF THIS IS USELESS
        # self.agent[-1].setParams(int(duration_sec * self.env.SIM_FREQ)//self.aggr_phy_steps, START)

        # Initializing action place holder with action at timestep_0 (t0)
        action = [
            self.agent[i].choose_action(self._preprocesObservationTraining(obs, i))
            for i in range(len(self.agent))
        ]
        action_dict = {
            str(i): np.append(action[i], [0, 1]) for i in range(len(self.agent))
        }

        # Setting params and initializing clock for simulation
        START = time.time()
        # self.agent[-1].setParams(int(duration_sec * self.env.SIM_FREQ), START)

        # Main simulation loop
        for i in range(0, int(duration_sec * self.env.SIM_FREQ), self.aggr_phy_steps):

            # Checking completion of the episode
            if done:
                print("Crash")
                break

            # Acting in environment
            next_obs, rewards, done, _ = self.env.step(action_dict)

            if self.mode == "Train":
                # Populating the Reply buffer
                for agent in range(len(self.agent)):
                    self.agent[agent].remember(
                        self._preprocesObservationTraining(obs, agent),
                        action[agent],
                        rewards[agent],
                        self._preprocesObservationTraining(next_obs, agent),
                        int(done),
                    )

                # Learning step for agents
                for agent in self.agent:
                    a_loss, c_loss = agent.learn()

                A_loss += a_loss
                C_loss += c_loss
            # Score updating
            for agent in range(self.training_drones):
                score[agent] += rewards[agent]

            # Updating observations. it needs to be updated before computing next action
            obs = next_obs

            # Drawing heading ( NOT IMPLEMENTD YET )
            if self.debugHeadigs and self.heading:
                for j in range(self.num_drones):
                    self.env._debugHeadingsNeural(j, self.agent[j].getHeading())

            # Computing new actions
            if i % self.cntrl_freq == 0:
                action = [
                    self.agent[agent].choose_action(
                        self._preprocesObservationTraining(obs, agent)
                    )
                    for agent in range(len(self.agent))
                ]
                action_dict = {
                    str(i): np.append(action[i], [0, 1]) for i in range(len(self.agent))
                }

            if self.gui:
                sync(i, START, self.env.TIMESTEP)
        return score, A_loss, C_loss

    ################################################################################

    def close(self):
        self.env.close()

    ################################################################################

    def reset(self):
        obs = self.env.reset()
        return obs

    ################################################################################

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
