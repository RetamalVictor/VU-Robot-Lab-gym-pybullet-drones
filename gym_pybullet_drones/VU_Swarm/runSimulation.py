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
from gym_pybullet_drones.VU_Swarm.envs.VectorForceAviary import VectorForceAviary
from gym_pybullet_drones.VU_Swarm.envs.RLTestBedAviary import RLTestBedAviary
from gym_pybullet_drones.VU_Swarm.controllers.flockControl import Flock
from gym_pybullet_drones.VU_Swarm.controllers.neuralControl import NeuralControl
from gym_pybullet_drones.VU_Swarm.models.DDPG.ddpg_agent import Agent

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    ActionType,
    ObservationType,
)

DEFAULT_DRONE = DroneModel("cf2x")
DEFAULT_DRONE_NUMBER = 2
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_PLOT = False
DEFAULT_AGGREGATE = True
# DEFAULT_OBSTACLES = False
DEFAULT_SIMULATION_FREQ_HZ = 120
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 50
DEFAULT_OUTPUT_FOLDER = r"\results"
DEFAULT_COLAB = False
DEFAULT_MODE = "Flocking"  # "NeuralControl" or "Flocking"


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
):

    assert mode in ["NeuralControl", "Flocking"]
    INIT_XYZS = np.array(
        [
            [np.random.uniform(1, 3), np.random.uniform(1, 3), 1]
            for i in range(num_drones)
        ]
    )

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
    args["debug"] = False
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

        env = RLTestBedAviary(
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
        agent = Agent(
            alpha=0.000025,
            beta=0.00025,
            input_dims=[2],
            tau=0.001,
            env=env,
            batch_size=64,
            layer1_size=400,
            layer2_size=300,
            n_actions=2,
        )
        controller = NeuralControl(env, agent, args, logger=logger)
        #### Initialize the Simulation #################################
        controller.init()
        controller.train(5000)
        controller.testAgent(25)

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
