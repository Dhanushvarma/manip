import mujoco as mj
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
            
        # Find joints
        if joint_names is None:
            # We need to collect all joints from the model
            self._joints = []
            
            # Recursive function to find all joints in a body subtree
            def collect_joints(body):
                # Get the first joint in this body
                joint = body.first_joint()
                while joint:
                    self._joints.append(joint)
                    joint = body.next_joint(joint)
                
                # Process child bodies recursively
                child = body.first_body()
                while child:
                    collect_joints(child)
                    child = body.next_body(child)
            
            # Start collection from worldbody
            collect_joints(self._spec.worldbody)
        else:
            # Find specific joints by name
            self._joints = []
            
            # Helper function to find a joint by name
            def find_joint(body, name):
                # Check joints in this body
                joint = body.first_joint()
                while joint:
                    if joint.name == name:
                        return joint
                    joint = body.next_joint(joint)
                
                # Check in child bodies
                child = body.first_body()
                while child:
                    result = find_joint(child, name)
                    if result:
                        return result
                    child = body.next_body(child)
                
                return None
            
            # Find all specified joints
            for joint_name in joint_names:
                joint = find_joint(self._spec.worldbody, joint_name)
                if joint:
                    self._joints.append(joint)
                else:
                    raise ValueError(f"Joint '{joint_name}' not found in model")
        
        # Find the end effector and attachment sites
        self._eef_site = None
        self._attachment_site = None
        
        # Helper function to find a site by name
        def find_site(body, name):
            # Check sites in this body
            site = body.first_site()
            while site:
                if site.name == name:
                    return site
                site = body.next_site(site)
            
            # Check in child bodies
            child = body.first_body()
            while child:
                result = find_site(child, name)
                if result:
                    return result
                child = body.next_body(child)
            
            return None
        
        # Find the required sites
        self._eef_site = find_site(self._spec.worldbody, eef_site_name)
        self._attachment_site = find_site(self._spec.worldbody, attachment_site_name)
        
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
            child: The child spec to attach
            pos: Position offset
            quat: Quaternion orientation offset
            
        Returns:
            The attachment frame
        """
        # Create a frame at the attachment site with the specified offset
        frame = self._attachment_site.add_frame(pos=pos, quat=quat)
        
        # Get the first body in the child spec (assuming it's the root body)
        if hasattr(child, 'worldbody'):
            # If child is a spec
            child_body = child.worldbody.first_body()
        else:
            # If child is already a body
            child_body = child
        
        # Attach the body to the frame
        attached_body = frame.attach_body(child_body, "", "")
        
        return attached_body
    
    def get_eef_pose(self, model, data):
        """
        Get the end effector pose from MuJoCo data.
        
        Args:
            model: MuJoCo model object
            data: MuJoCo data object
            
        Returns:
            Numpy array [pos, quat] of the end effector pose
        """
        # Get the site ID from its name
        site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, self._eef_site.name)
        
        # Get position directly from data
        ee_pos = data.site_xpos[site_id].copy()
        
        # Get orientation matrix and convert to quaternion
        site_xmat = data.site_xmat[site_id].reshape(3, 3)
        ee_quat = np.zeros(4)
        mj.mju_mat2Quat(ee_quat, site_xmat)
        
        # Combine into a single pose array
        ee_pose = np.concatenate((ee_pos, ee_quat))
        
        return ee_pose