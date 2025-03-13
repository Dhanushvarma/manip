import numpy as np
import mujoco


class JointEffortController:
    def __init__(
        self,
        model,
        data,
        joints,
        min_effort: np.ndarray,
        max_effort: np.ndarray,
    ) -> None:
        self._model = model
        self._data = data
        self._joint_names = joints
        self._min_effort = min_effort
        self._max_effort = max_effort

        # Get joint IDs for faster lookup
        self._joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name.name)
            for joint_name in self._joint_names
        ]

    def run(self, target) -> None:
        """
        Run the robot controller.
        Parameters:
            target (numpy.ndarray): The desired target joint efforts for the robot.
            The size of `target` should be (n_joints,) where n_joints is the number of robot joints.
        """
        # Clip the target efforts to ensure they are within the allowable effort range
        target_effort = np.clip(target, self._min_effort, self._max_effort)

        # Set the applied forces for each joint

        _dof_indexes = []
        for i, joint_id in enumerate(self._joint_ids):
            dof_index = self._model.jnt_dofadr[joint_id]
            _dof_indexes.append(dof_index)      
        
        
        self._data.qfrc_applied[_dof_indexes] = target_effort

    def reset(self) -> None:
        pass
