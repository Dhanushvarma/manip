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
        
        # Add a site to visualize the target
        # For sphere geometry, need size with 3 elements (only first is used but API requires 3)
        self._site = self._mocap.add_site(
            name="target_site",
            type=mj.mjtGeom.mjGEOM_SPHERE,
            size=[0.02, 0, 0],  # For sphere, only first value is used but need 3 elements
            rgba=[1, 0, 0, 0.5]
        )
        
        # Store the body name for later use
        self.body_name = self._mocap.name
        
    def set_mocap_pose(self, physics, position, quaternion):
        """Set the pose of the mocap body.
        
        Args:
            physics: MuJoCo physics/data object.
            position: Target position.
            quaternion: Target orientation as quaternion.
        """
        # Get the body ID
        body_id = mj.mj_name2id(physics.model, mj.mjtObj.mjOBJ_BODY, self.body_name)
        
        # Get the mocap ID for this body
        if physics.model.body_mocapid[body_id] != -1:
            mocap_id = physics.model.body_mocapid[body_id]
            
            # Set the mocap pose
            physics.data.mocap_pos[mocap_id] = position
            physics.data.mocap_quat[mocap_id] = quaternion
    
    def get_mocap_pose(self, physics):
        """Get the current pose of the mocap body.
        
        Args:
            physics: MuJoCo physics/data object.
        
        Returns:
            pose: 7-element array with [position, quaternion]
        """
        # Get the body ID
        body_id = mj.mj_name2id(physics.model, mj.mjtObj.mjOBJ_BODY, self.body_name)
        
        # Get the mocap ID for this body
        if physics.model.body_mocapid[body_id] != -1:
            mocap_id = physics.model.body_mocapid[body_id]
            
            # Get the mocap pose
            position = physics.data.mocap_pos[mocap_id].copy()
            quaternion = physics.data.mocap_quat[mocap_id].copy()
            
            return np.concatenate([position, quaternion])
        
        return None