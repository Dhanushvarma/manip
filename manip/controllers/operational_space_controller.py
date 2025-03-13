from manip.controllers import JointEffortController
import numpy as np
import mujoco
from manip.utils.controller_utils import (
    task_space_inertia_matrix,
    pose_error,
)
from manip.utils.transform_utils import mat2quat

np.set_printoptions(precision=5, threshold=5, edgeitems=3, suppress=True)


class OperationalSpaceController(JointEffortController):
    def __init__(
        self,
        model,
        data,
        joints,
        eef_site,
        min_effort: np.ndarray,
        max_effort: np.ndarray,
        kp: float,
        ko: float,
        kv: float,
        vmax_xyz: float,
        vmax_abg: float,
    ) -> None:
        super().__init__(model, data, joints, min_effort, max_effort)

        self._model = model
        self._data = data
        self._eef_site = eef_site
        self._kp = kp
        self._ko = ko
        self._kv = kv
        self._vmax_xyz = vmax_xyz
        self._vmax_abg = vmax_abg

        # Get eef site ID
        self._eef_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, eef_site.name)

        # Get joint IDs and DOF addresses
        self._joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint.name)
            for joint in joints
        ]
        self._jnt_dof_ids = [model.jnt_dofadr[jid] for jid in self._joint_ids]
        self._dof = len(self._jnt_dof_ids)

        self._task_space_gains = np.array([self._kp] * 3 + [self._ko] * 3)
        self._lamb = self._task_space_gains / self._kv
        self._sat_gain_xyz = vmax_xyz / self._kp * self._kv
        self._sat_gain_abg = vmax_abg / self._ko * self._kv
        self._scale_xyz = vmax_xyz / self._kp * self._kv
        self._scale_abg = vmax_abg / self._ko * self._kv

    def run(self, target, grip=None):
        # target is a 7D vector [x, y, z, qx, qy, qz, qw]
        target_pose = target[[0, 1, 2, 4, 5, 6, 3]]

        # Get the Jacobian matrix for the end-effector
        jacp = np.zeros((3, self._model.nv))
        jacr = np.zeros((3, self._model.nv))
        mujoco.mj_jacSite(self._model, self._data, jacp, jacr, self._eef_id)
        J = np.vstack((jacp, jacr))
        J = J[:, self._jnt_dof_ids]

        # Get the mass matrix for the controlled degrees of freedom (DOF)
        M_full = np.zeros((self._model.nv, self._model.nv))
        mujoco.mj_fullM(self._model, M_full, self._data.qM)
        # M = M_full[np.ix_(self._jnt_dof_ids, self._jnt_dof_ids)]
        M = M_full[self._jnt_dof_ids, :][:, self._jnt_dof_ids]
        Mx, M_inv = task_space_inertia_matrix(M, J)

        # Get the joint velocities for the controlled DOF
        dq = np.array([self._data.qvel[dof_id] for dof_id in self._jnt_dof_ids])

        # Get the end-effector position and orientation
        ee_pos = self._data.site_xpos[self._eef_id].copy()
        site_xmat = self._data.site_xmat[self._eef_id].reshape(3, 3)
        ee_quat = mat2quat(site_xmat)
        ee_pose = np.concatenate([ee_pos, ee_quat])

        # Calculate the pose error (difference between target and current pose)
        pose_err = pose_error(target_pose, ee_pose)

        # Initialize the task space control signal
        u_task = np.zeros(6)

        # Calculate the task space control signal
        u_task += self._scale_signal_vel_limited(pose_err)

        # Joint space control signal
        u = np.zeros(self._dof)

        # Add the task space control signal to the joint space control signal
        u += np.dot(J.T, np.dot(Mx, u_task))

        # Add damping to joint space control signal
        u += -self._kv * np.dot(M, dq)

        # Add gravity compensation to joint space control signal
        u += self._data.qfrc_bias[self._jnt_dof_ids]

        # HACK(dhanush): WidowX gripper control
        if grip is not None:
            # NOTE(dhanush): +- order gives open | -+ order gives close
            grip_torque = 1.0
            # open
            if grip == 0:
                u[-2] = grip_torque
                u[-1] = -grip_torque
            # close
            elif grip == 1:
                u[-2] = -grip_torque
                u[-1] = grip_torque

        # Send the target effort to the joint effort controller
        super().run(u)

    def _scale_signal_vel_limited(self, u_task: np.ndarray) -> np.ndarray:
        """
        Scale the control signal such that the arm isn't driven to move faster in position or orientation than the specified vmax values.

        Parameters:
            u_task (numpy.ndarray): The task space control signal.

        Returns:
            numpy.ndarray: The scaled task space control signal.
        """
        norm_xyz = np.linalg.norm(u_task[:3])
        norm_abg = np.linalg.norm(u_task[3:])
        scale = np.ones(6)
        if norm_xyz > self._sat_gain_xyz:
            scale[:3] *= self._scale_xyz / norm_xyz
        if norm_abg > self._sat_gain_abg:
            scale[3:] *= self._scale_abg / norm_abg

        return self._kv * scale * self._lamb * u_task
