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

class SingleController:
    """
    TO-DO:
    - extract take off and landing to a parent class
    - Learning method. Include the trajectory colletion inside this class
    - Plotting not only one reward, but also average.

    - The observation does not include the neigbors. So they just avoid the target to not collide.
    """

    def __init__(self, env, agent, args, checkpoint=False, mode="Eval"):
        """
        TO-DO:
        [ ] Describe the parameters here
        """
        assert mode in ["Train", "Eval"]
        self.checkpoint = checkpoint
        self.env = env
        self.dt = 1 / env.SIM_FREQ
        self.cntrl_freq = args["cntrl_freq"]
        self.aggr_phy_steps = args["aggr_phy_steps"]
        self.num_drones = args["num_drones"]
        self.num_close_drones = args["NNeigh"]
        self.gui = args["gui"]
        self.training_drones = 1
        exp_name = args["exp_name"]
        self.logging_dir = os.path.join(
            args["logging_dir"], os.path.join("tensorboard", f"Experiment_{exp_name}l")
        )

        if not os.path.exists(self.logging_dir):
            os.makedirs(self.logging_dir)
        self.agent = agent
        self.action = {str(i): np.array([0, 0, 0, 1]) for i in range(self.num_drones)}
        self.mode = mode

        for agent in range(self.training_drones):
            self.agent[agent].initHeading()
        
        if self.checkpoint:
            for agent in range(self.training_drones):
                self.agent[agent].load_models()

################################################################################

    def train(self, episodes, duration_sec=10):
        """
        Main training method
        """
        # Initializing tensorboard writer
        self.writer = SummaryWriter(self.logging_dir)

        # Main training loop (Now, cap to num of episodes)
        for i in range(episodes):
            # Collect trajectory
            print(f"Episode: {i}")
            START = time.time()
            episode_score, A_loss, C_loss = self.trainingLoop(duration_sec)
            print(f"Episode {i} duration: {time.time() - START}")


            # Information about training
            if i % 5 == 0:
                pprint(episode_score)
                print()

            # Saving best models
            if self.mode == "Train":
                # self.writer.add_scalar("Agent 1 reward", episode_score[0], i)
                self.writer.add_scalar("Actor loss", A_loss, i)
                self.writer.add_scalar("Critic loss", C_loss, i)
                self.writer.add_scalar("Mean reward", np.mean(episode_score), i)
                if i % (episodes//5) == 0:
                    self.agent[0].save_models()

        self.writer.close()

    ################################################################################

    def _preprocesObservationTraining(self, obs, nth_drone):
        """
        This is deprecated with the addition of _preprocessObs in the env class
        """
        return obs

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
                print("Target reached")
                break

            # Acting in environment
            next_obs, rewards, done, _ = self.env.step(action_dict)

            if self.mode == "Train":
                # Populating the Reply buffer
                for agent in range(1):
                    self.agent[agent].remember(
                        self._preprocesObservationTraining(obs, agent),
                        action[agent],
                        rewards,
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
                score[agent] += rewards

            # Updating observations. it needs to be updated before computing next action
            obs = next_obs

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
                self.env._debugHeadingsNeural(0, self.agent[0].heading)
                sync(i, START, self.env.TIMESTEP)
                
        return score, A_loss, C_loss

    ################################################################################

    def close(self):
        self.env.close()

    ################################################################################

    def reset(self):
        self.env.resetTarget()
        obs = self.env.reset()
        print(self.env.target)
        return obs

