import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from multiagent_env import VectorForceAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    ActionType,
    ObservationType,
)


DEFAULT_DRONE = DroneModel("cf2x")
DEFAULT_DRONE_NUMBER = 2
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_PLOT = False
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_AGGREGATE = True
DEFAULT_OBSTACLES = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 20
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_COLAB = False


def run(
    drone=DEFAULT_DRONE,
    num_drones=DEFAULT_DRONE_NUMBER,
    gui=DEFAULT_GUI,
    record_video=DEFAULT_RECORD_VIDEO,
    plot=DEFAULT_PLOT,
    aggregate=DEFAULT_AGGREGATE,
    simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
    control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
    duration_sec=DEFAULT_DURATION_SEC,
    output_folder=DEFAULT_OUTPUT_FOLDER,
    colab=DEFAULT_COLAB,
):

    #### Initialize the simulation #############################
    INIT_XYZS = np.array(
        [
            [0, 0, 0.1],
            [0.3, 0, 0.01],
        ]
    )
    INIT_RPYS = np.array(
        [
            [0, 0, 0],
            [0, 0, np.pi / 3],
        ]
    )
    AGGR_PHY_STEPS = int(simulation_freq_hz / control_freq_hz) if aggregate else 1
    PHY = Physics.PYB

    #### Create the environment ################################
    env = VectorForceAviary(
        drone_model=drone,
        num_drones=num_drones,
        neighbourhood_radius=10,
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        physics=Physics.PYB,
        freq=simulation_freq_hz,
        aggregate_phy_steps=AGGR_PHY_STEPS,
        gui=gui,
        record=record_video,
        obs=ObservationType.KIN,
        act=ActionType.VEL,
    )

    #### Compute number of control steps in the simlation ######

    #### Initialize the velocity target ########################
    #### Initialize the logger #################################
    logger = Logger(
        logging_freq_hz=int(simulation_freq_hz / AGGR_PHY_STEPS),
        num_drones=num_drones,
        output_folder=output_folder,
        colab=colab,
    )

    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ / control_freq_hz))
    action = {str(i): np.array([0, 0, 0, 0]) for i in range(num_drones)}

    START = time.time()
    for i in range(0, int(duration_sec * env.SIM_FREQ) // 8, AGGR_PHY_STEPS):

        for j in range(num_drones):
            env._showDroneLocalAxes(j)

        obs, reward, done, info = env.step(action)

        if i % CTRL_EVERY_N_STEPS == 0:
            for j in range(num_drones):
                action[str(j)] = np.array([0, 0, 1, j + 1])

        if i % env.SIM_FREQ == 0:
            env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.TIMESTEP)
    START = time.time()
    for i in range(0, int(duration_sec * env.SIM_FREQ) // 4, AGGR_PHY_STEPS):

        for j in range(num_drones):
            env._showDroneLocalAxes(j)

        obs, reward, done, info = env.step(action)

        if i % CTRL_EVERY_N_STEPS == 0:
            for j in range(num_drones):
                action[str(j)] = np.array([1, 0, 0, j + 1])

        if i % env.SIM_FREQ == 0:
            env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.TIMESTEP)

    START = time.time()
    for i in range(0, int(duration_sec * env.SIM_FREQ) // 4, AGGR_PHY_STEPS):

        for j in range(num_drones):
            env._showDroneLocalAxes(j)

        obs, reward, done, info = env.step(action)

        if i % CTRL_EVERY_N_STEPS == 0:
            for j in range(num_drones):
                action[str(j)] = np.array([0, -1, 0, j + 1])

        if i % env.SIM_FREQ == 0:
            env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.TIMESTEP)

    START = time.time()
    for i in range(0, int(duration_sec * env.SIM_FREQ) // 4, AGGR_PHY_STEPS):

        for j in range(num_drones):
            env._showDroneLocalAxes(j)

        obs, reward, done, info = env.step(action)

        if i % CTRL_EVERY_N_STEPS == 0:
            for j in range(num_drones):
                action[str(j)] = np.array([-1, 0, 0, j + 1])

        if i % env.SIM_FREQ == 0:
            env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.TIMESTEP)
    #### Close the environment #################################
    START = time.time()
    for i in range(0, int(duration_sec * env.SIM_FREQ) // 4, AGGR_PHY_STEPS):

        for j in range(num_drones):
            env._showDroneLocalAxes(j)

        obs, reward, done, info = env.step(action)

        if i % CTRL_EVERY_N_STEPS == 0:
            for j in range(num_drones):
                action[str(j)] = np.array([0, 1, 0, j + 1])

        if i % env.SIM_FREQ == 0:
            env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.TIMESTEP)
    env.close()

    #### Plot the simulation results ###########################
    # logger.save_as_csv("vel") # Optional CSV save
    # if plot:
    #     logger.plot()


if __name__ == "__main__":

    run()
