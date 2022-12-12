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

from envs.VectorForceAviary import VectorForceAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    ActionType,
    ObservationType,
)


DEFAULT_DRONE = DroneModel("cf2x")
DEFAULT_DRONE_NUMBER = 6
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_PLOT = False
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_AGGREGATE = True
DEFAULT_OBSTACLES = False
DEFAULT_SIMULATION_FREQ_HZ = 120
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 45
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_COLAB = False


def wraptopi(x):
    x = x % (3.1415926 * 2)
    x = (x + (3.1415926 * 2)) % (3.1415926 * 2)
    x[x > 3.1415926] = x[x > 3.1415926] - (3.1415926 * 2)
    return x


def avg_heading_angle(headings):
    """
    returns the angle from a sum of headings in radians
    """
    headings = headings[headings != 0]
    X = np.sum(np.cos(headings))
    Y = np.sum(np.sin(headings))
    return np.arctan2(Y, X)


# def avg_heading_angle_y(headings):
#     headings = headings[headings != 0]
#     X = np.sum(np.cos(headings))
#     Y = np.sum(np.sin(headings))
#     h_Y = Y / np.sqrt((X*X) + (Y*Y))

#     return h_Y
# def avg_heading_angle_x(headings):
#     headings = headings[headings != 0]
#     X = np.sum(np.cos(headings))
#     Y = np.sum(np.sin(headings))
#     h_X = X / np.sqrt((X*X) + (Y*Y))
#     return h_X


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
            [np.random.uniform(-2, 2), np.random.uniform(-2, 2), 0.1]
            for i in range(num_drones)
        ]
    )
    INIT_RPYS = np.array([[0, 0, 0] for i in range(num_drones)])

    AGGR_PHY_STEPS = int(simulation_freq_hz / control_freq_hz) if aggregate else 1
    PHY = Physics.PYB
    EPSILON = 2
    SIGMA = 0.8
    umax_const = 0.4
    wmax = 1.5708 * 2
    k1 = 1
    k2 = 0.5
    dt = 1 / DEFAULT_SIMULATION_FREQ_HZ
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

    #### Initialize the logger #################################
    logger = Logger(
        logging_freq_hz=int(simulation_freq_hz / AGGR_PHY_STEPS),
        num_drones=num_drones,
        output_folder=output_folder,
        colab=colab,
    )
    action = {str(i): np.array([0, 0, 0, 0.5]) for i in range(num_drones)}
    START = time.time()
    for i in range(1, int(5 * env.SIM_FREQ), AGGR_PHY_STEPS):

        obs, reward, done, info = env.step(action)

        if i % env.SIM_FREQ == 0:
            env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.TIMESTEP)
    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ / control_freq_hz))
    action = {str(i): np.array([0, 0, 1, 1]) for i in range(num_drones)}
    TARGET_H = 0.75
    START = time.time()
    for i in range(1, int(2 * env.SIM_FREQ), AGGR_PHY_STEPS):

        # for j in range(num_drones): env._showDroneLocalAxes(j)

        obs, reward, done, info = env.step(action)
        if env._getDroneStateVector(0)[2] == TARGET_H:
            break
        if i % CTRL_EVERY_N_STEPS == 0:
            for j in range(num_drones):
                action[str(j)] = np.array([0, 0, 1, 1 / i])

        if i % env.SIM_FREQ == 0:
            env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.TIMESTEP)

    action = {str(i): np.array([0, 0, 0, 1]) for i in range(num_drones)}
    positions = np.zeros((2, num_drones))
    START = time.time()
    # headings = np.array([np.pi for i in range(num_drones)])
    headings = np.array([np.random.uniform(-np.pi, np.pi) for i in range(num_drones)])
    for i in range(0, int(duration_sec * env.SIM_FREQ), AGGR_PHY_STEPS):

        ############################################################

        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)
        # print(f'Reward for drones = {reward}')
        #### Compute control at the desired frequency ##############
        if i % CTRL_EVERY_N_STEPS == 0:

            for j in range(num_drones):
                state = env._getDroneStateVector(j)
                positions[0][j] = state[0]
                positions[1][j] = state[1]

            X1, XT = np.meshgrid(positions[0], positions[0])
            Y1, YT = np.meshgrid(positions[1], positions[1])
            H1, HT = np.meshgrid(headings, headings)

            D_ij_x = X1 - XT
            D_ij_y = Y1 - YT

            D_ij = np.sqrt(np.multiply(D_ij_x, D_ij_x) + np.multiply(D_ij_y, D_ij_y))
            D_ij[(D_ij >= 1) | (D_ij == 0)] = np.inf

            # now all the headings are always present
            headings_avg = avg_heading_angle(headings)

            # for j in range(num_drones): env._debugHeadingsAverage(j,headings_avg - headings)
            for j in range(num_drones):
                env._debugHeadings(j, headings)

            Bearnig_angles = np.arctan2(D_ij_y, D_ij_x)
            Bearnig_angles_local = (
                Bearnig_angles - HT + headings * np.identity(num_drones)
            )

            forces = -(EPSILON) * ((SIGMA**4 / D_ij**5) - (SIGMA**2 / D_ij**3))

            forces[D_ij == np.inf] = 0.0
            forces = np.nan_to_num(forces)

            alpha, beta = 2, 1
            p_x = alpha * np.sum(forces * np.cos(Bearnig_angles_local), axis=1)
            p_y = alpha * np.sum(forces * np.sin(Bearnig_angles_local), axis=1)

            h_x = beta * np.cos(headings_avg - headings)
            h_y = beta * np.sin(headings_avg - headings)

            fx = p_x + h_x
            fy = p_y + h_y

            U = fx * k1
            U[U > umax_const] = umax_const
            U[U < 0] = 0.005

            w = fy * k2
            w[w > wmax] = wmax
            w[w < -wmax] = -wmax

            vx = U * np.cos(headings) * dt
            vy = U * np.sin(headings) * dt
            headings = wraptopi(headings + w * dt)

            for j in range(num_drones):
                action[str(j)] = np.array([vx[j], vy[j], 0, 1])
                # action[str(j)][0] = vx[j]
        ### Log the simulation ####################################
        for j in range(num_drones):
            logger.log(
                drone=j,
                timestamp=i / env.SIM_FREQ,
                state=obs[str(j)]["state"],
            )

        #### Printout ##############################################
        if i % env.SIM_FREQ == 0:
            env.render()

        if i % 500 == 0:
            print(action)
            print(headings)
            print(D_ij)

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Plot the simulation results ###########################
    # logger.save_as_csv("vel") # Optional CSV save
    if plot:
        logger.plot()


if __name__ == "__main__":

    run()
