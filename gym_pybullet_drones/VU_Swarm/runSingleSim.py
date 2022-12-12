import numpy as np
import os

import sys

sys.path.append(r'C:\Users\victo\Desktop\VU master\drones\Drones_RL\gym-pybullet-drones')

from gym_pybullet_drones.VU_Swarm.envs.SingleDroneAviary import SingleDroneAviary
from gym_pybullet_drones.VU_Swarm.models.DDPG.ddpg_agent import Agent
from gym_pybullet_drones.VU_Swarm.models.DDPG.ddpg_agent_heading import AgentHeading
from gym_pybullet_drones.VU_Swarm.controllers.SingleControl import SingleController
from torch.utils.tensorboard import SummaryWriter

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    ActionType,
    ObservationType,
)
def run():
    DEFAULT_DRONE = DroneModel("cf2x")
    DEFAULT_DRONE_NUMBER = 2
    DEFAULT_GUI = False
    DEFAULT_AGGREGATE = True
    DEFAULT_SIMULATION_FREQ_HZ = 120
    DEFAULT_CONTROL_FREQ_HZ = 48
    EXP_NAME = "Exp_single_2_heading_8hdl"

    # INIT_XYZS = np.array(
    #     [
    #         [np.random.uniform(-2.5, 2.5), np.random.uniform(-3, 2.5), 1]
    #         for _ in range(DEFAULT_DRONE_NUMBER)
    #     ]
    # )
    # INIT_XYZS = np.array([[0, i, 1] for i in range(DEFAULT_DRONE_NUMBER)])
    INIT_XYZS = np.array([[0, 0, 1], [0,0,0]])
    INIT_RPYS = np.array([[0, 0, 0] for i in range(DEFAULT_DRONE_NUMBER)])

    AGGR_PHY_STEPS = int(DEFAULT_SIMULATION_FREQ_HZ / DEFAULT_CONTROL_FREQ_HZ) if DEFAULT_AGGREGATE else 1

    args = {}
    args["gui"] = DEFAULT_GUI
    args["idle_time"] = 1
    args["cntrl_freq"] = int(np.floor(DEFAULT_SIMULATION_FREQ_HZ / DEFAULT_CONTROL_FREQ_HZ))
    args["aggr_phy_steps"] = AGGR_PHY_STEPS
    args["num_drones"] = DEFAULT_DRONE_NUMBER
    args["target"] = [0, 0]
    args["umax_cont"] = 0.4
    args["wmax"] = 1.5708 * 2
    args["debug"] = True
    args["NNeigh"] = 3
    args["exp_name"] = EXP_NAME
    args[
        "logging_dir"
    ] = rf"C:\Users\victo\Desktop\VU master\drones\Drones_RL\gym-pybullet-drones\gym_pybullet_drones\VU_Swarm\experiments\{EXP_NAME}"
    if not os.path.exists(args["logging_dir"]):
        os.makedirs(args["logging_dir"])


    env = SingleDroneAviary(initial_xyzs=INIT_XYZS, initial_rpys=INIT_RPYS, gui=DEFAULT_GUI)
    
    agent_list = [
                AgentHeading(
                    alpha=3e-3,
                    beta=1e-3,
                    input_dims=[4],
                    tau=3e-4,
                    checkpoint_dir=args["logging_dir"],
                    dt=0.005,
                    max_size=1000000,
                    batch_size=64,  # 64
                    layer1_size=12,
                    layer2_size=8,
                    n_actions=2,
                )
                for _ in range(1)    
    ]

    controller = SingleController(
            env, agent_list, args, checkpoint=False, mode="Train"
        )
        #### Initialize the Simulation #################################
        # controller.init()
    controller.train(250, 15)

if __name__ == '__main__':

    run()
