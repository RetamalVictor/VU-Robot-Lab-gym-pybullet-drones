#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Victor Retamal
# Last modification Date: 07/10/2022
#  - Victor Retamal --
# version ='1.0'
# ---------------------------------------------------------------------------
""" Main Simulation Script """
# ---------------------------------------------------------------------------


import numpy as np
import os
import sys

sys.path.append(r'C:\Users\victo\Desktop\VU master\drones\Drones_RL\gym-pybullet-drones')
from gym_pybullet_drones.VU_Swarm.envs.VectorForceAviary import VectorForceAviary
from gym_pybullet_drones.VU_Swarm.envs.MARLNeuralFlockingAviary import (
    MARLNeuralFlockingAviary,
)
from gym_pybullet_drones.VU_Swarm.controllers.flockControl import Flock
from gym_pybullet_drones.VU_Swarm.controllers.neuralControlMarl import NeuralControl
from gym_pybullet_drones.VU_Swarm.models.DDPG.ddpg_agent import Agent
from gym_pybullet_drones.VU_Swarm.models.DDPG.ddpg_agent_heading import AgentHeading
from gym_pybullet_drones.VU_Swarm.models.cicularModel import CircularAgent

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    ActionType,
    ObservationType,
)

DEFAULT_DRONE = DroneModel("cf2x")
DEFAULT_DRONE_NUMBER = 5
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_PLOT = False
DEFAULT_AGGREGATE = True
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 120
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 50
DEFAULT_OUTPUT_FOLDER = r"\results"
DEFAULT_COLAB = False
DEFAULT_MODE = "Flocking"  # "NeuralControl" or "Flocking"
DEFAULT_HEADING = True
EXP_NAME = "Exp_heading"


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
    mode=DEFAULT_MODE,
    heading=DEFAULT_HEADING,
):

    assert mode in ["NeuralControl", "Flocking"]
    INIT_XYZS = np.array(
        [
            [np.random.uniform(-2.5, 2.5), np.random.uniform(-3, 2.5), 1]
            for _ in range(num_drones)
        ]
    )

    # INIT_XYZS = np.array([[1,3,1],
    #                     [2,2,1],
    #                     [-1,-2,1],
    #                     [3,1,1],
    #                     [1,1,1]])
    INIT_RPYS = np.array([[0, 0, 0] for i in range(num_drones)])

    AGGR_PHY_STEPS = int(simulation_freq_hz / control_freq_hz) if aggregate else 1

    #### Initialize the logger #################################
    logger = Logger(
        logging_freq_hz=int(simulation_freq_hz / AGGR_PHY_STEPS),
        num_drones=num_drones,
        output_folder=output_folder,
        colab=colab,
    )
    args = {}
    args["gui"] = gui
    args["idle_time"] = 1
    args["cntrl_freq"] = int(np.floor(simulation_freq_hz / control_freq_hz))
    args["aggr_phy_steps"] = AGGR_PHY_STEPS
    args["num_drones"] = num_drones
    args["target"] = [0, 0]
    args["sensing_distance"] = 4.5
    args["epsilon"] = 2
    args["sigma"] = 1
    args["umax_cont"] = 0.4
    args["wmax"] = 1.5708 * 2
    args["k1"] = 1
    args["k2"] = 0.5
    args["alpha"] = 2
    args["beta"] = 5
    args["debug"] = True
    args["heading"] = heading
    args["NNeigh"] = 3
    args["exp_name"] = EXP_NAME
    args[
        "logging_dir"
    ] = rf"C:\Users\victo\Desktop\VU master\drones\Drones_RL\gym-pybullet-drones\gym_pybullet_drones\VU_Swarm\experiments\{EXP_NAME}"
    if not os.path.exists(args["logging_dir"]):
        os.makedirs(args["logging_dir"])

    #### Create the environment ################################
    """
    For the environment, you can change the physic engine. The options are:
    PYB = "pyb"                         # Base PyBullet physics update
    DYN = "dyn"                         # Update with an explicit model of the dynamics
    PYB_GND = "pyb_gnd"                 # PyBullet physics update with ground effect
    PYB_DRAG = "pyb_drag"               # PyBullet physics update with drag
    PYB_DW = "pyb_dw"                   # PyBullet physics update with downwash
    PYB_GND_DRAG_DW = "pyb_gnd_drag_dw" # PyBullet physics update with ground effect, drag, and downwash
    """
    if mode == "NeuralControl":

        env = MARLNeuralFlockingAviary(
            drone_model=drone,
            num_drones=num_drones,
            neighbourhood_radius=10,
            initial_xyzs=INIT_XYZS,
            initial_rpys=INIT_RPYS,
            physics=Physics.PYB_DRAG,
            freq=simulation_freq_hz,
            aggregate_phy_steps=AGGR_PHY_STEPS,
            gui=gui,
            record=record_video,
            obs=ObservationType.KIN,
            act=ActionType.VEL,
        )
        if heading == True:
            agent_list = [
                AgentHeading(
                    alpha=0.000025,
                    beta=0.00025,
                    input_dims=[num_drones],
                    tau=0.001,
                    checkpoint_dir=args["logging_dir"],
                    max_size=3000000,
                    batch_size=64,  # 64
                    layer1_size=400,
                    layer2_size=300,
                    n_actions=2,
                    control_frq=args["cntrl_freq"],
                )
                for _ in range(num_drones)
            ]

        else:
            agent_list = [
                Agent(
                    alpha=0.000025,
                    beta=0.00025,
                    input_dims=[num_drones],
                    tau=0.001,
                    checkpoint_dir=args["logging_dir"],
                    max_size=3000000,
                    batch_size=64,  # 64
                    layer1_size=400,
                    layer2_size=300,
                    n_actions=2,
                )
                for _ in range(num_drones)
            ]

        # target_agent = CircularAgent(radius=1.5, kPosition=5, center_circle=[0,0],env=env)
        controller = NeuralControl(
            env, agent_list, args, checkpoint=False, mode="Train", logger=logger
        )
        #### Initialize the Simulation #################################
        # controller.init()
        controller.train(200, 15)
        # controller.testAgent(25)

        if plot:
            controller.plot()
        controller.close()

    elif mode == "Flocking":

        env = VectorForceAviary(
            drone_model=drone,
            num_drones=num_drones,
            neighbourhood_radius=10,
            initial_xyzs=INIT_XYZS,
            initial_rpys=INIT_RPYS,
            physics=Physics.PYB_DRAG,
            freq=simulation_freq_hz,
            aggregate_phy_steps=AGGR_PHY_STEPS,
            gui=gui,
            record=record_video,
            obs=ObservationType.KIN,
            act=ActionType.VEL,
        )
        #### Initialize the flock #################################
        flocking = Flock(env, args, logger)

        flocking.init()
        flocking.takeOff(target_height=1, duration=3)
        flocking.flockLoop(duration_sec=duration_sec)

        if plot:
            flocking.plot()
        flocking.close()


if __name__ == "__main__":

    run()
