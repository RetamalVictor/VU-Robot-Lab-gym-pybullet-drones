import os
from typing import Tuple
import numpy as np
from gym import spaces
import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.multi_agent_rl.BaseMultiagentAviary import (
    BaseMultiagentAviary,
)
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    ActionType,
    ObservationType,
)


class RLTestBedAviary(BaseMultiagentAviary):
    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        num_drones: int = 2,
        neighbourhood_radius: float = np.inf,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        freq: int = 240,
        aggregate_phy_steps: int = 1,
        gui=True,
        record=False,
        obs: ObservationType = ObservationType.KIN,
        act: ActionType = ActionType.VEL,
        target: Tuple[int, int] = [0, 0],
    ):

        #### Create integrated controllers #########################
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
        if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
            self.ctrl = [
                DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)
            ]
        elif drone_model == DroneModel.HB:
            raise ValueError(
                "[ERROR] in VelocityAviary.__init__(), velocity control not supported for DroneModel.HB."
            )

        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            neighbourhood_radius=neighbourhood_radius,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            freq=freq,
            aggregate_phy_steps=aggregate_phy_steps,
            gui=gui,
            record=record,
            obs=obs,
            act=act,
        )
        #### Set a limit on the maximum target speed ###############
        self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000 / 3600)
        self.target = np.array(target)

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        dict[str, ndarray]
            A Dict of Box(4,) with NUM_DRONES entries,
            indexed by drone Id in string format.

        """
        #### Action vector ######### X       Y       Z   fract. of MAX_SPEED_KMH
        act_lower_bound = np.array([-1, -1, -1, 0])
        act_upper_bound = np.array([1, 1, 1, 1])
        return spaces.Dict(
            {
                str(i): spaces.Box(
                    low=act_lower_bound, high=act_upper_bound, dtype=np.float32
                )
                for i in range(self.NUM_DRONES)
            }
        )

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        dict[str, dict[str, ndarray]]
            A Dict with NUM_DRONES entries indexed by Id in string format,
            each a Dict in the form {Box(20,), MultiBinary(NUM_DRONES)}.

        """
        #### Observation vector ### X Y Z Q1 Q2 Q3 Q4 R P Y VX VY VZ WX WY WZ P0 P1 P2 P3
        obs_lower_bound = np.array(
            [
                -np.inf,  # X
                -np.inf,  # Y
                0.0,  # Z
                -1.0,  # Q1
                -1.0,  # Q2
                -1.0,  # Q3
                -1.0,  # Q4
                -np.pi,  # R
                -np.pi,  # P
                -np.pi,  # Yw
                -np.inf,  # VX
                -np.inf,  # VY
                -np.inf,  # VZ
                -np.inf,  # WX
                -np.inf,  # WY
                -np.inf,  # WZ
                0.0,  # P0
                0.0,  # P1
                0.0,  # P2
                0.0,  # P3
            ]
        )
        obs_upper_bound = np.array(
            [
                np.inf,
                np.inf,
                np.inf,
                1.0,
                1.0,
                1.0,
                1.0,
                np.pi,
                np.pi,
                np.pi,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                self.MAX_RPM,
                self.MAX_RPM,
                self.MAX_RPM,
                self.MAX_RPM,
            ]
        )
        return spaces.Dict(
            {
                str(i): spaces.Dict(
                    {
                        "state": spaces.Box(
                            low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32
                        ),
                        "neighbors": spaces.MultiBinary(self.NUM_DRONES),
                    }
                )
                for i in range(self.NUM_DRONES)
            }
        )

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        For the value of key "state", see the implementation of `_getDroneStateVector()`,
        the value of key "neighbors" is the drone's own row of the adjacency matrix.

        Returns
        -------
        dict[str, dict[str, ndarray]]
            A Dict with NUM_DRONES entries indexed by Id in string format,
            each a Dict in the form {Box(20,), MultiBinary(NUM_DRONES)}.

        """
        # Remeber to update this later
        # adjacency_mat = self._getAdjacencyMatrix()
        return self._getDroneStateVector(0)[:2]
        # return {
        #     str(i): {
        #         "state": self._getDroneStateVector(i),
        #         "neighbors": adjacency_mat[i, :],
        #     }
        #     for i in range(self.NUM_DRONES)
        # }

    ################################################################################

    def _preprocessAction(self, action):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Uses PID control to target a desired velocity vector.
        Converts a dictionary into a 2D array.

        Parameters
        ----------
        action : dict[str, ndarray]
            The desired velocity input for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        rpm = np.zeros((self.NUM_DRONES, 4))
        for k, v in action.items():
            #### Get the current state of the drone  ###################
            state = self._getDroneStateVector(int(k))
            #### Normalize the first 3 components of the target velocity
            if np.linalg.norm(v[0:3]) != 0:
                v_unit_vector = v[0:3] / np.linalg.norm(v[0:3])
            else:
                v_unit_vector = np.zeros(3)
            temp, _, _ = self.ctrl[int(k)].computeControl(
                control_timestep=self.AGGR_PHY_STEPS * self.TIMESTEP,
                cur_pos=state[0:3],
                cur_quat=state[3:7],
                cur_vel=state[10:13],
                cur_ang_vel=state[13:16],
                target_pos=state[0:3],  # same as the current position
                target_rpy=np.array([0, 0, state[9]]),  # keep current yaw
                # target_rpy=np.array([0,0,v[4]]), # keep current yaw
                target_vel=self.SPEED_LIMIT
                * np.abs(v[3])
                * v_unit_vector,  # target the desired velocity vector
            )
            rpm[int(k), :] = temp
        return rpm

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Returns
        -------
        dict[int, float]
            The reward value for each drone.

        """
        # obs here is dictionary of the form {"i":{"state": Box(20,), "neighbors": MultiBinary(NUM_DRONES)}}
        # parse velocity and position
        states = self._getDroneStateVector(0)

        # rewards[0] =  -np.linalg.norm(states[0, 0:2] - self.target) ** 2
        # rewards[1] = - np.linalg.norm(states[0, 0:2] - states[1, 0:2]) ** 2

        return -np.linalg.norm(states[0:2] - self.target) ** 2

    ################################################################################

    def _computeDone(self):
        """Computes the current done value(s).

        Returns
        -------
        dict[int | "__all__", bool]
            Dictionary with the done value of each drone and
            one additional boolean value for key "__all__".

        """
        # bool_val = (
        #     True if self.step_counter / self.SIM_FREQ > self.EPISODE_LEN_SEC else False
        # )
        # done = {i: bool_val for i in range(self.NUM_DRONES)}
        # done["__all__"] = True if True in done.values() else False

        states = self._getDroneStateVector(0)
        return np.linalg.norm(states[0:2] - self.target) ** 2 < 0.2

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[int, dict[]]
            Dictionary of empty dictionaries.

        """
        return {i: {} for i in range(self.NUM_DRONES)}

    ################################################################################

    def _clipAndNormalizeState(self, state):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY * self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z * self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi  # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(
                state,
                clipped_pos_xy,
                clipped_pos_z,
                clipped_rp,
                clipped_vel_xy,
                clipped_vel_z,
            )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi  # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = (
            state[13:16] / np.linalg.norm(state[13:16])
            if np.linalg.norm(state[13:16]) != 0
            else state[13:16]
        )

        norm_and_clipped = np.hstack(
            [
                normalized_pos_xy,
                normalized_pos_z,
                state[3:7],
                normalized_rp,
                normalized_y,
                normalized_vel_xy,
                normalized_vel_z,
                normalized_ang_vel,
                state[16:20],
            ]
        ).reshape(
            20,
        )

        return norm_and_clipped

    ################################################################################

    def _clipAndNormalizeStateWarning(
        self,
        state,
        clipped_pos_xy,
        clipped_pos_z,
        clipped_rp,
        clipped_vel_xy,
        clipped_vel_z,
    ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.

        """
        if not (clipped_pos_xy == np.array(state[0:2])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in FlockAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(
                    state[0], state[1]
                ),
            )
        if not (clipped_pos_z == np.array(state[2])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in FlockAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(
                    state[2]
                ),
            )
        if not (clipped_rp == np.array(state[7:9])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in FlockAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(
                    state[7], state[8]
                ),
            )
        if not (clipped_vel_xy == np.array(state[10:12])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in FlockAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(
                    state[10], state[11]
                ),
            )
        if not (clipped_vel_z == np.array(state[12])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in FlockAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(
                    state[12]
                ),
            )

    ################################################################################

    def _debugHeadings(self, nth_drone, headings):
        AXIS_LENGTH = 2 * self.L
        self.X_AX[nth_drone] = p.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=[
                np.cos(headings[nth_drone]) / 5,
                np.sin(headings[nth_drone]) / 5,
                0,
            ],
            lineColorRGB=[1, 0, 0],
            parentObjectUniqueId=self.DRONE_IDS[nth_drone],
            parentLinkIndex=-1,
            replaceItemUniqueId=int(self.X_AX[nth_drone]),
            physicsClientId=self.CLIENT,
        )

    ################################################################################

    def _debugHeadingsAverage(self, nth_drone, headings):
        self.Y_AX[nth_drone] = p.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=[
                np.sin(headings[nth_drone]) / 5,
                np.cos(headings[nth_drone]) / 5,
                0,
            ],
            lineColorRGB=[0, 1, 0],
            parentObjectUniqueId=self.DRONE_IDS[nth_drone],
            parentLinkIndex=-1,
            replaceItemUniqueId=int(self.X_AX[nth_drone]),
            physicsClientId=self.CLIENT,
        )

    ################################################################################
