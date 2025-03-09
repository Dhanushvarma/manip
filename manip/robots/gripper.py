import mujoco as mj


class Gripper:
    def __init__(self, xml_path, joint_name, actuator_name, name: str = None):
        """
        Initialize a Gripper from an XML file.

        Args:
            xml_path: Path to the XML file
            joint_name: Name of the gripper's control joint
            actuator_name: Name of the gripper's actuator
            name: Optional model name
        """
        self._spec = mj.MjSpec.from_file(xml_path)

        if name:
            self._spec.modelname = name

        # Find joint by name
        self._joint = None
        for joint in self._spec.find_all(mj.mjtObj.mjOBJ_JOINT):
            if joint.name == joint_name:
                self._joint = joint
                break

        if not self._joint:
            raise ValueError(f"Joint '{joint_name}' not found in model")

        # Find all bodies
        self._bodies = self._spec.find_all(mj.mjtObj.mjOBJ_BODY)

        # Find actuator by name
        self._actuator = None
        for actuator in self._spec.find_all(mj.mjtObj.mjOBJ_ACTUATOR):
            if actuator.name == actuator_name:
                self._actuator = actuator
                break

        if not self._actuator:
            raise ValueError(f"Actuator '{actuator_name}' not found in model")

    @property
    def joint(self):
        """Joint element for controlling the gripper."""
        return self._joint

    @property
    def actuator(self):
        """Actuator element for the gripper."""
        return self._actuator

    @property
    def spec(self):
        """Returns the MjSpec object corresponding to this gripper."""
        return self._spec
