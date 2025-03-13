import mujoco as mj
import numpy as np


class Target:
    def __init__(self, spec):
        """Initialize a target.

        Args:
            spec: MjSpec model to which this target will be added.
        """
        self._spec = spec

        # Add mocap body to the worldbody
        self._mocap = self._spec.worldbody.add_body(name="mocap", mocap=True)
        self._site = self._mocap.add_geom(
            name="target",
            type=mj.mjtGeom.mjGEOM_BOX,
            rgba=[1, 0, 0, 0.25],
            size=[0.015] * 3,
            conaffinity=0,
            contype=0,
        )

        # Store the body name for later use
        self.body_name = self._mocap.name

    @property
    def spce(self) -> object:
        return self._spec

    @property
    def mocap(self) -> object:
        return self._mocap

    def set_mocap_pose(self, data, position, quaternion):
        data.mocap_pos[0] = position
        data.mocap_quat[0] = quaternion
        pass

    def get_mocap_pose(self, data):
        position = data.mocap_pos[0]
        quaternion = data.mocap_quat[0]
        pose = np.concatenate([position, quaternion])
        return pose
