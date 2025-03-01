import numpy as np

# Access to enums and MuJoCo library functions.

# PyMJCF
from dm_control import mjcf

# Composer high level imports
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.manipulation.shared.observations import (
    ObservableNames,
    make_options,
)


_WIDOW_ARM_XML_PATH = "/home/dhanush/dhanush_ws/lira/manip/assets/mujoco_menagerie/trossen_vx300s/vx300s.xml"

_JOINT_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
    "left_finger",
    # "right_finger"   # NOTE(dhanush): commented as right finger is not associated with any actuator
]

_ACTUATOR_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
    "finger",
]

_VELOCITY_LIMITS = {k: np.pi for k in _JOINT_NAMES}
_WRIST_SITE = "pinch"

from dm_control.entities.manipulators import base
from dm_control.composer import define


class WidowArm(base.RobotArm):
    """A composer entity representing a Widow arm."""

    def _build(self, name=None):
        """Initializes the WidowArm.

        Args:
          name: String, the name of this robot. Used as a prefix in the MJCF name
            name attributes.
        """

        self._mjcf_root = mjcf.from_path(_WIDOW_ARM_XML_PATH)
        # self._mjcf_root.model =

        if name:
            self._mjcf_root.model = name
        # Find MJCF elements that will be exposed as attributes.
        self._joints = [self._mjcf_root.find("joint", name) for name in _JOINT_NAMES]
        self._wrist_site = self._mjcf_root.find("site", _WRIST_SITE)
        self._bodies = self.mjcf_model.find_all("body")

        self._actuators = [
            self._mjcf_root.find("actuator", name) for name in _ACTUATOR_NAMES
        ]

        # NOTE: not required for now.(1, 0, 0, 1)
        if False:
            # Add torque sensors.
            self._joint_torque_sensors = [
                _add_torque_sensor(joint) for joint in self._joints
            ]

    def _build_observables(self):
        return WidowArmObservables(self)

    @property
    def joints(self):
        """List of joint elements belonging to the arm."""
        return self._joints

    @property
    def actuators(self):
        """List of actuator elements belonging to the arm."""
        return self._actuators

    @property
    def wrist_site(self):
        """Wrist site of the arm (attachment point for the hand)."""
        return self._wrist_site

    @property
    def mjcf_model(self):
        """Returns the `mjcf.RootElement` object corresponding to this robot."""
        return self._mjcf_root


# NOTE: not required for now.
if False:

    @property
    def joint_torque_sensors(self):
        """List of torque sensors for each joint belonging to the arm."""
        return self._joint_torque_sensors

    def _add_torque_sensor(joint):
        """Adds a torque sensor to a joint, returns the new MJCF element."""
        site = joint.parent.add(
            "site",
            size=[1e-3],
            group=composer.SENSOR_SITES_GROUP,
            name=joint.name + "_site",
        )
        return joint.root.sensor.add("torque", site=site, name=joint.name + "_torque")


class WidowArmObservables(base.JointsObservables):
    """Jaco arm obserables."""

    # TODO: check if Transpose is required for the joint_pos
    @define.observable
    def joints_pos(self):
        # Because most of the Jaco arm joints are unlimited, we return the joint
        # angles as sine/cosine pairs so that the observations are bounded.
        def joint_angles(physics):
            joint_pos = physics.bind(self._entity.joints).qpos
            return joint_pos

        return observable.Generic(joint_angles)

    @define.observable
    def joints_vel(self):
        def joint_velocities(physics):
            joint_vel = physics.bind(self._entity.joints).qvel
            return joint_vel

        return observable.Generic(joint_velocities)

    # NOTE(dhanush) : Not Required as of now.
    if False:

        @define.observable
        def joints_torque(self):
            # MuJoCo's torque sensors are 3-axis, but we are only interested in torques
            # acting about the axis of rotation of the joint. We therefore project the
            # torques onto the joint axis.
            def get_torques(physics):
                torques = physics.bind(self._entity.joint_torque_sensors).sensordata
                joint_axes = physics.bind(self._entity.joints).axis
                return np.einsum("ij,ij->i", torques.reshape(-1, 3), joint_axes)

            return observable.Generic(get_torques)


WIDOW_ARM_OBSERVABLES = ObservableNames(proprio=["joints_pos", "joints_vel"])


# Custom fucntion to make WidowArm
def make_widow_arm(obs_settings):
    """Constructs a robot arm with manipulation-specific defaults.

    Args:
      obs_settings: `observations.ObservationSettings` instance.

    Returns:
      An instance of `manipulators.base.RobotArm`.
    """
    return WidowArm(
        observable_options=make_options(obs_settings, WIDOW_ARM_OBSERVABLES)
    )
