import numpy as np

from dm_control.manipulation.shared import cameras
from dm_control.composer import initializers
from dm_control.manipulation.shared import constants
from dm_control.manipulation.shared import workspaces
from dm_control.utils import rewards
from dm_control.composer.variation import distributions
from dm_control import composer
from dm_control.manipulation.shared import registry
from dm_control.manipulation.shared import tags

from mjmanip.props.colorbox import ColorBoxWithSites
from mjmanip.robots.widow import make_widow_arm
from mjmanip.utils.arenas import CustomArena
from mjmanip.utils.cameras import FRONT_FLAT_FAR

_DISTANCE_TO_LIFT = 0.25


class MyLift(composer.Task):
    """A task where the goal is to elevate a prop."""

    def __init__(
        self,
        arena,
        arm,
        blocks,
        obs_settings,
        workspace,
        control_timestep,
        target_block_index=None,
    ):
        """Initializes a new `Lift` task.

        Args:
            arena: `composer.Entity` instance.
            arm: `robot_base.RobotArm` instance.
            blocks: List of `composer.Entity` instances.
            obs_settings: `observations.ObservationSettings` instance.
            workspace: `_LiftWorkspace` specifying the placement of the prop and TCP.
            control_timestep: Float specifying the control timestep in seconds.
            target_block_index: Integer index of the target block to lift or None if any block can be lifted.
        """
        self._arena = arena
        self._arm = arm
        self._arena.attach_offset(self._arm, offset=workspace.arm_offset)
        self.control_timestep = control_timestep
        self._task_observables = cameras.add_camera_observables(
            arena, obs_settings, FRONT_FLAT_FAR
        )
        self._blocks = blocks
        self._target_block_index = target_block_index

        for block_ in blocks:
            self._arena.add_free_entity(block_)

        self._prop_placer = initializers.PropPlacer(
            props=blocks,
            position=distributions.Uniform(*workspace.prop_bbox),
            quaternion=workspaces.uniform_z_rotation,
            ignore_collisions=True,  # NOTE: hmm?
            settle_physics=True,
        )

        self._target_height_site = workspaces.add_bbox_site(
            body=self.root_entity.mjcf_model.worldbody,
            lower=(-1, -1, 0),
            upper=(1, 1, 0),
            rgba=constants.RED,
            name="target_height",
        )

        # TODO: iffy about this, check!
        workspaces.add_bbox_site(
            body=self.root_entity.mjcf_model.worldbody,
            lower=workspace.prop_bbox.lower,
            upper=workspace.prop_bbox.upper,
            rgba=constants.BLUE,
            name="prop_spawn_area",
        )

    @property
    def root_entity(self):
        return self._arena

    @property
    def arm(self):
        return self._arm

    @property
    def task_observables(self):
        return self._task_observables

    def _get_height_of_lowest_vertex(self, physics):
        min_height = np.inf

        if self._target_block_index is not None:
            # Only consider the target block if index is specified
            target_block = self._blocks[self._target_block_index]
            for corner_site in target_block.corner_sites:
                min_height = np.minimum(
                    min_height, physics.bind(corner_site).xpos[0, 2]
                )
        else:
            # Consider all blocks if no specific target
            for block in self._blocks:
                for corner_site in block.corner_sites:
                    min_height = np.minimum(
                        min_height, physics.bind(corner_site).xpos[2]
                    )

        return min_height

    def get_reward(self, physics):
        prop_height = self._get_height_of_lowest_vertex(physics)
        return rewards.tolerance(
            prop_height,
            bounds=(self._target_height, np.inf),
            margin=_DISTANCE_TO_LIFT,
            value_at_margin=0,
            sigmoid="linear",
        )

    def initialize_episode(self, physics, random_state):
        self._prop_placer(physics, random_state)
        initial_prop_height = self._get_height_of_lowest_vertex(physics)
        self._target_height = _DISTANCE_TO_LIFT + initial_prop_height
        physics.bind(self._target_height_site).pos[2] = self._target_height

    def before_step(self, physics, action, random_state):
        return super().before_step(physics, action, random_state)
    
    def before_substep(self, physics, action, random_state):
        return super().before_substep(physics, action, random_state)


import collections
import itertools

from dm_control.manipulation.shared import arenas
from dm_control.manipulation.shared import constants
from dm_control.manipulation.shared import observations

WIDOW_ARM_OFFSET = (
    0.0,
    0.4,
    0.0,
    0.7071068,
    0,
    0,
    -0.7071068,
)  # MODIFY THIS as per reachability
_BOX_SIZE = 0.0225
_BOX_MASS = 0.1

_MyLiftWorkspace = collections.namedtuple(
    "_MyLiftWorkspace", ["prop_bbox", "arm_offset"]
)


_BOX_WORKSPACE = _MyLiftWorkspace(
    prop_bbox=workspaces.BoundingBox(
        lower=(-0.1, -0.1, _BOX_SIZE), upper=(0.1, 0.1, _BOX_SIZE)
    ),
    arm_offset=WIDOW_ARM_OFFSET,
)


# _build(self, half_lengths=None, mass=None, name="box", rgba=[1, 0, 0, 1.0]):


def _mylift(obs_settings):

    arena = CustomArena()
    arm = make_widow_arm(obs_settings=obs_settings)

    # workspace configuration
    workspace = _BOX_WORKSPACE

    # blocks
    red_block = ColorBoxWithSites(
        half_lengths=[_BOX_SIZE] * 3,
        rgba=(1, 0, 0, 1),
        mass=_BOX_MASS,
        observable_options=observations.make_options(
            obs_settings, observations.FREEPROP_OBSERVABLES
        ),
    )

    green_block = ColorBoxWithSites(
        half_lengths=[_BOX_SIZE] * 3,
        rgba=(0, 1, 0, 1),
        mass=_BOX_MASS,
        observable_options=observations.make_options(
            obs_settings, observations.FREEPROP_OBSERVABLES
        ),
    )

    blocks = [red_block, green_block]

    task = MyLift(
        arena=arena,
        arm=arm,
        blocks=blocks,
        obs_settings=obs_settings,
        workspace=workspace,
        control_timestep=constants.CONTROL_TIMESTEP,
        target_block_index=None,
    )

    return task


# Modifications to increase the camera size
from dm_control.manipulation.shared.observations import (
    ObservationSettings,
    _ENABLED_FTT,
    _DISABLED_FEATURE,
    _ENABLED_FEATURE,
    CameraObservableSpec,
)


_LARGE_ENABLED_CAMERA = CameraObservableSpec(
    height=300,
    width=300,
    enabled=True,
    update_interval=1,
    buffer_size=1,
    delay=0,
    aggregator=None,
    corruptor=None,
)

print("Modified")

LARGE_VISION = ObservationSettings(
    proprio=_ENABLED_FEATURE,
    ftt=_ENABLED_FTT,
    prop_pose=_DISABLED_FEATURE,
    camera=_LARGE_ENABLED_CAMERA,
)


@registry.add(tags.VISION)
def my_lift_vision():
    return _mylift(obs_settings=LARGE_VISION)
