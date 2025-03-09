import mujoco as mj
from manip.utils.transform_utils import mat2quat
import numpy as np


class Arm:
    def __init__(
        self,
        xml_path,
        eef_site_name,
        attachment_site_name,
        joint_names=None,
        name: str = None,
    ):
        """
        Initialize an Arm from an XML file.

        Args:
            xml_path: Path to the XML file
            eef_site_name: Name of the end effector site
            attachment_site_name: Name of the attachment site
            joint_names: Optional list of joint names to use
            name: Optional model name
        """
        self._spec = mj.MjSpec.from_file(xml_path)

        if name:
            self._spec.modelname = name

        # Find MJCF elements that will be exposed as attributes
        if joint_names is None:
            self._joints = self._spec.find_all(mj.mjtObj.mjOBJ_JOINT)
        else:
            self._joints = []
            for name in joint_names:
                joint = None
                for j in self._spec.find_all(mj.mjtObj.mjOBJ_JOINT):
                    if j.name == name:
                        joint = j
                        break
                if joint:
                    self._joints.append(joint)
                else:
                    raise ValueError(f"Joint '{name}' not found in model")

        # Find sites by name
        sites = self._spec.find_all(mj.mjtObj.mjOBJ_SITE)
        self._eef_site = None
        self._attachment_site = None

        for site in sites:
            if site.name == eef_site_name:
                self._eef_site = site
            if site.name == attachment_site_name:
                self._attachment_site = site

        if not self._eef_site:
            raise ValueError(f"End effector site '{eef_site_name}' not found")
        if not self._attachment_site:
            raise ValueError(f"Attachment site '{attachment_site_name}' not found")

    @property
    def joints(self):
        """List of joint elements belonging to the arm."""
        return self._joints

    @property
    def eef_site(self):
        """Wrist site of the arm (attachment point for the hand)."""
        return self._eef_site

    @property
    def spec(self):
        """Returns the MjSpec object corresponding to this robot."""
        return self._spec

    def attach_tool(self, child, pos=[0, 0, 0], quat=[1, 0, 0, 0]):
        """
        Attach a tool to the arm's attachment site.

        Args:
            child: The child body to attach
            pos: Position offset
            quat: Quaternion orientation offset

        Returns:
            The attachment frame
        """
        # Create a frame at the attachment site with the specified offset
        frame = self._attachment_site.add_frame(pos=pos, quat=quat)

        # Attach the child to this frame
        # Note: We're assuming child has a body that can be attached
        result = frame.attach_body(child, "", "")

        return result

    def get_eef_pose(self, physics):
        """
        Get the end effector pose from physics.

        Args:
            physics: MuJoCo physics/data object

        Returns:
            Numpy array [pos, quat] of the end effector pose
        """
        # Bind the site to physics data
        site_data = physics.bind(self._eef_site)

        # Get position and orientation
        ee_pos = site_data.xpos
        ee_quat = mat2quat(site_data.xmat.reshape(3, 3))

        # Combine into a single pose array
        ee_pose = np.concatenate((ee_pos, ee_quat))

        return ee_pose
