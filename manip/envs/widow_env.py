import time
import os

import numpy as np
from loguru import logger
import gymnasium as gym
from gymnasium import spaces

import mujoco
import mujoco.viewer
from manip.arenas import StandardArena
from manip.robots import Arm
from manip.props import Primitive
from manip.mocaps import Target
from manip.controllers import OperationalSpaceController
from manip.utils.consts import RED, GREEN


class WidowEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,  # Set a proper render FPS
    }

    def __init__(self, render_mode=None):
        """
        Initialize the WidowEnv environment.

        Args:
            render_mode (str, optional): Rendering mode, either "human" or "rgb_array".
        """

        # set all env relevant constants
        self._init_consts()

        # Observation space with reasonable image dimensions
        self.observation_space = spaces.Dict(
            {
                "pinch_site_pose": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64
                ),  # pos(3) + quat(4)
                "joint_pose": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64
                ),  # 8 joint positions
                "frontview_image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self._image_height, self._image_width, 3),
                    dtype=np.uint8,
                ),  # RGB image with useful dimensions
                "topview_image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self._image_height, self._image_width, 3),
                    dtype=np.uint8,
                ),  # RGB image with useful dimensions
            }
        )

        # Action space for operational space control (dx, dy, dz, droll, dpitch, dyaw, grip)
        continuous_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float64)
        discrete_space = spaces.Discrete(2)
        self.action_space = spaces.Tuple((continuous_space, discrete_space))

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self._render_mode = render_mode

        ############################
        # create MJCF model
        ############################

        # checkerboard floor
        self._arena = StandardArena()

        # mocap target that OSC will try to follow
        self._target = Target(self._arena.spec)

        # Widow Arm
        self._arm = Arm(
            xml_path=os.path.join(
                os.path.dirname(__file__),
                "../assets/robots/trossen_wx250s/wx250s.xml",
            ),
            eef_site_name="pinch_site",
            attachment_site_name="attachment_site",
        )

        # table
        self._table = Primitive(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=self._table_dims,
            pos=[0, 0, self._table_dims[-1]],
            rgba=[1, 1, 0, 1],
            friction=[1, 0.3, 0.0001],
        )

        # red box
        self._red_box = Primitive(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[self._cube_prop_size] * 3,
            pos=[0, 0, self._cube_prop_size],
            rgba=RED,
            friction=[1, 0.3, 0.0001],
            mass=0.01,
        )
        self._green_box = Primitive(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[self._cube_prop_size] * 3,
            pos=[0, 0, self._cube_prop_size],
            rgba=GREEN,
            friction=[1, 0.3, 0.0001],
            mass=0.01,
        )

        # attach arm, table, boxes
        self._arena.attach(self._arm.spec, pos=self._robot_root_pose)
        self._arena.attach(self._table.spec, pos=self._table_root_pose)
        self._arena.attach_free(
            self._red_box.spec,
            pos=[
                self._table_root_pose[0],
                self._cube_y_offset,
                2 * self._table_dims[2],
            ],
        )
        self._arena.attach_free(
            self._green_box.spec,
            pos=[
                self._table_root_pose[0],
                -self._cube_y_offset,
                2 * self._table_dims[2],
            ],
        )

        # attach cameras
        self._arena.attach_camera(
            name="frontview", pos=[1.15, 0, 0.2125], quat=[0.5, 0.5, 0.5, 0.5]
        )
        self._arena.attach_camera(
            name="topview", pos=[0.38, 0, 0.8], quat=[0.7071068, 0, 0, 0.7071068]
        )

        # generate model
        self._model = self._arena.spec.compile()
        self._data = mujoco.MjData(self._model)

        # increase camera buffer size
        self._model.vis.global_.offwidth = self._image_width
        self._model.vis.global_.offheight = self._image_height

        # set up OSC controller with appropriate parameters for Widow Arm
        self._controller = OperationalSpaceController(
            model=self._model,
            data=self._data,
            joints=self._arm.joints,
            eef_site=self._arm.eef_site,
            min_effort=-25.0,
            max_effort=25.0,
            kp=self._controller_gains["kp"],
            ko=self._controller_gains["ko"],
            kv=self._controller_gains["kv"],
            vmax_xyz=1.0,
            vmax_abg=1.0,
        )

        # for time keeping
        self._timestep = self._model.opt.timestep
        self._step_start = None
        self._viewer = None

        # Track task completion
        self._task_success = False

    def _init_consts(self):
        """
        All constant relevant to setting up the task
        """
        self._controller_gains = {"kv": 50, "kp": 200, "ko": 200}
        self._robot_rest_joint_cfg = [0, 0, 0, 0, +1.5708, 0, 0.015, -0.015]

        self._image_width = 224
        self._image_height = 224

        self._table_dims = [0.15, 0.3, 0.075]
        self._cube_prop_size = 0.015

        self._robot_root_pose = [0, 0, 0]
        self._table_root_pose = [0.325, 0, 0]

        self._cube_y_offset = 0.15
        self._box_spawn_bounds = [0.20, 0.20, 0]  # xyz from root pose

        # Define the height threshold for successful block lifting
        self._lift_height_threshold = 0.08  # 8cm above table surface

        # Table top surface height from ground
        self._table_top_height = self._table_dims[-1] * 2

    def _get_obs(self) -> dict:
        # Get pinch site pose (position and orientation)
        site_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, self._arm.eef_site.name
        )
        pinch_site_pos = self._data.site_xpos[site_id].copy()
        pinch_site_mat = self._data.site_xmat[site_id]
        pinch_site_quat = np.zeros(4)
        mujoco.mju_mat2Quat(pinch_site_quat, pinch_site_mat)

        pinch_site_pose = np.concatenate([pinch_site_pos, pinch_site_quat])

        # Get joint positions
        joint_pose = np.zeros(len(self._arm.joints))
        for i, joint in enumerate(self._arm.joints):
            joint_id = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_JOINT, joint.name
            )
            joint_pose[i] = self._data.qpos[joint_id]

        # Render actual camera images
        frontview_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_CAMERA, "frontview"
        )
        topview_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_CAMERA, "topview"
        )

        # Setup renderer
        renderer = mujoco.Renderer(self._model, self._image_height, self._image_width)

        # Render views
        renderer.update_scene(self._data, camera=frontview_id)
        # frontview_image = renderer.render()
        frontview_image = np.zeros((self._image_height, self._image_width, 3))

        renderer.update_scene(self._data, camera=topview_id)
        # topview_image = renderer.render()
        topview_image = np.zeros((self._image_height, self._image_width, 3))

        # Store the frontview image for rendering
        self._current_frame = frontview_image

        return {
            "pinch_site_pose": pinch_site_pose,
            "joint_pose": joint_pose,
            "frontview_image": frontview_image,
            "topview_image": topview_image,
        }

    def _get_info(self) -> dict:
        # Get block positions
        red_box_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_GEOM, self._red_box.geom.name
        )
        green_box_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_GEOM, self._green_box.geom.name
        )

        red_box_pos = self._data.geom_xpos[red_box_id].copy()
        green_box_pos = self._data.geom_xpos[green_box_id].copy()

        # Calculate height above table
        red_box_height_above_table = red_box_pos[2] - self._table_top_height
        green_box_height_above_table = green_box_pos[2] - self._table_top_height

        # Check if either block is above threshold
        red_box_lifted = red_box_height_above_table > self._lift_height_threshold
        green_box_lifted = green_box_height_above_table > self._lift_height_threshold

        # Get end-effector position
        site_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, self._arm.eef_site.name
        )
        ee_position = self._data.site_xpos[site_id].copy()

        # Provide useful information for debugging and monitoring
        info = {
            "red_box_pos": red_box_pos,
            "green_box_pos": green_box_pos,
            "red_box_height_above_table": red_box_height_above_table,
            "green_box_height_above_table": green_box_height_above_table,
            "red_box_lifted": red_box_lifted,
            "green_box_lifted": green_box_lifted,
            "task_success": self._task_success,
            "ee_position": ee_position,
        }
        return info

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)

        print("Reset is occurring!")

        # Use the seed for randomization
        self.np_random = np.random.RandomState(seed)

        # Reset MuJoCo data
        mujoco.mj_resetData(self._model, self._data)

        for i, joint in enumerate(self._arm.joints):
            joint_id = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_JOINT, joint.name
            )
            self._data.qpos[joint_id] = self._robot_rest_joint_cfg[i]

        # randomize box positions
        red_box_pos = [
            self.np_random.uniform(
                self._table_root_pose[0] - self._box_spawn_bounds[0],
                self._table_root_pose[0] + self._box_spawn_bounds[0],
            ),
            self.np_random.uniform(
                self._cube_y_offset - self._box_spawn_bounds[1],
                self._cube_y_offset + self._box_spawn_bounds[1],
            ),
            2 * self._table_dims[2],
        ]
        green_box_pos = [
            self.np_random.uniform(
                self._table_root_pose[0] - self._box_spawn_bounds[0],
                self._table_root_pose[0] + self._box_spawn_bounds[0],
            ),
            self.np_random.uniform(
                -self._cube_y_offset - self._box_spawn_bounds[1],
                -self._cube_y_offset + self._box_spawn_bounds[1],
            ),
            2 * self._table_dims[2],
        ]

        # Fix: Setting box positions using body position instead of geom
        red_box_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, f"{self._red_box.geom.name}_body"
        )
        green_box_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, f"{self._green_box.geom.name}_body"
        )

        # Set positions in qpos for the free bodies
        qpos_adr = self._model.body_jntadr[red_box_body_id]
        self._data.qpos[qpos_adr : qpos_adr + 3] = red_box_pos
        self._data.qpos[qpos_adr + 3 : qpos_adr + 7] = [
            1,
            0,
            0,
            0,
        ]  # Identity quaternion

        qpos_adr = self._model.body_jntadr[green_box_body_id]
        self._data.qpos[qpos_adr : qpos_adr + 3] = green_box_pos
        self._data.qpos[qpos_adr + 3 : qpos_adr + 7] = [
            1,
            0,
            0,
            0,
        ]  # Identity quaternion

        # Fix: Set target position above table
        target_pos = [0.5, 0, self._table_top_height + 0.1]
        target_quat = [0.7071068, 0, 0.7071068, 0]  # Y = 90 degrees

        # Get mocap body id for target
        target_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, self._target.body_name
        )

        # Set mocap position and orientation
        if self._model.body_mocapid[target_body_id] != -1:
            mocap_id = self._model.body_mocapid[target_body_id]
            self._data.mocap_pos[mocap_id] = target_pos
            self._data.mocap_quat[mocap_id] = target_quat

        # Forward kinematics to update all positions
        mujoco.mj_forward(self._model, self._data)

        # Reset task success flag
        self._task_success = False

        # Get initial observation
        observation = self._get_obs()

        # Render the initial state if in human mode
        if self._render_mode == "human":
            self._render_frame()

        info = self._get_info()

        return observation, info

    def step(self, action) -> tuple:
        # Unpack the action tuple - continuous and discrete parts
        continuous_action, grip = action

        # Use the action to update the target pose
        site_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, self._arm.eef_site.name
        )

        # Get current end-effector position
        current_ee_pos = self._data.site_xpos[site_id].copy()

        # Get current end-effector orientation (fix for quaternion issue)
        current_ee_mat = self._data.site_xmat[site_id]
        current_ee_quat = np.zeros(4)
        mujoco.mju_mat2Quat(current_ee_quat, current_ee_mat)

        # Scale the continuous action for position
        action_scale = 0.02  # 2cm per step
        position_delta = continuous_action[:3] * action_scale

        # Update position target based on action (dx, dy, dz)
        new_target_pos = current_ee_pos + position_delta

        # Clamp to reasonable workspace
        new_target_pos = np.clip(
            new_target_pos,
            [0.1, -0.5, 0.02],  # Lower bounds
            [0.7, 0.5, 0.5],  # Upper bounds
        )

        print("Current EE Pos: ", current_ee_pos)
        print("Current EE Quat: ", current_ee_quat)

        # Get target body id
        target_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, self._target.body_name
        )

        # Set mocap position and orientation
        if self._model.body_mocapid[target_body_id] != -1:
            mocap_id = self._model.body_mocapid[target_body_id]
            self._data.mocap_pos[mocap_id] = new_target_pos
            self._data.mocap_quat[mocap_id] = current_ee_quat

        # Run OSC controller to move to target pose
        target_pose = np.concatenate([new_target_pos, current_ee_quat])
        self._controller.run(target_pose, grip=grip)

        # Step physics
        mujoco.mj_step(self._model, self._data)

        # Get block positions and check if lifted
        info = self._get_info()

        # Task is successful if either block is lifted above threshold
        self._task_success = info["red_box_lifted"] or info["green_box_lifted"]

        # Binary reward: 1 for success, 0 otherwise
        reward = 1.0 if self._task_success else 0.0

        # Terminate episode if task is successful
        terminated = self._task_success

        # Get observation
        observation = self._get_obs()

        # Render frame if in human mode
        if self._render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self) -> np.ndarray:
        """
        Renders the current frame and returns it as an RGB array if the render mode is set to "rgb_array".

        Returns:
            np.ndarray: RGB array of the current frame.
        """
        if self._render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self) -> None:
        """
        Renders the current frame and updates the viewer if the render mode is set to "human".
        """
        if self._viewer is None and self._render_mode == "human":
            # launch viewer
            self._viewer = mujoco.viewer.launch_passive(
                self._model,
                self._data,
            )
        if self._step_start is None and self._render_mode == "human":
            # initialize step timer
            self._step_start = time.time()

        if self._render_mode == "human":
            # render viewer
            self._viewer.sync()

            # manage frame rate
            time_until_next_step = self._timestep - (time.time() - self._step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            self._step_start = time.time()

        else:  # rgb_array
            print("RGB")
            renderer = mujoco.Renderer(
                self._model, self._image_height, self._image_width
            )
            renderer.update_scene(self._data)
            return renderer.render()

    def close(self) -> None:
        """
        Closes the viewer if it's open.
        """
        if self._viewer is not None:
            self._viewer.close()
