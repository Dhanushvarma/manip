import mujoco as mj


class StandardArena:
    def __init__(self) -> None:
        """
        Initializes the StandardArena object by creating a new MjSpec model and adding a solid color floor and lights.
        """
        self._spec = mj.MjSpec()

        # Set options
        self._spec.option.timestep = 0.002
        self._spec.option.flag.warmstart = (
            True  # In mjSpec, use boolean instead of "enable"
        )

        # Create floor material (solid color)
        self._spec.add_material(
            name="grid",
            rgba=[0.3, 0.4, 0.5, 1.0],
            reflectance=0.2,
        )

        # Add floor
        self._spec.worldbody.add_geom(
            type=mj.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1], material="grid"
        )

        # Add skybox
        self._spec.add_texture(
            name="skybox",
            type=mj.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mj.mjtBuiltin.mjBUILTIN_GRADIENT,
            rgb1=[1, 1, 1],
            rgb2=[1, 1, 1],
            width=512,
            height=3072,
        )

        # Add lights
        for x in [-2, 2]:
            self._spec.worldbody.add_light(
                pos=[x, -1, 3], dir=[-x, 1, -2], castshadow=False
            )

        # Enable copying during attachment to handle namespace conflicts
        self._spec.copy_during_attach = True

    def attach(self, child, pos=[0, 0, 0], quat=[1, 0, 0, 0]):
        """
        Attaches a child element to the MJCF model at a specified position and orientation.

        Args:
            child: The child element to attach.
            pos: The position of the child element.
            quat: The orientation of the child element.

        Returns:
            The frame of the attached child element.
        """
        # Create a frame at the specified position and orientation
        frame = self._spec.worldbody.add_frame(pos=pos, quat=quat)

        # Attach the child to this frame
        # Note: We're assuming child is a body that can be attached
        frame.attach_body(child, "", "")

        return frame

    def attach_free(self, child, pos=[0, 0, 0], quat=[1, 0, 0, 0]):
        """
        Attaches a child element to the MJCF model with a free joint.

        Args:
            child: The child element to attach.
            pos: The position of the child element.
            quat: The orientation of the child element.

        Returns:
            The frame of the attached child element.
        """
        # First attach the child
        frame = self.attach(child, pos, quat)

        # Add a free joint to the first body of the attached element
        attached_body = frame.first_body()
        if attached_body:
            attached_body.add_freejoint()

        return frame

    @property
    def spec(self):
        """
        Returns the MjSpec for the StandardArena object.

        Returns:
            The MjSpec model.
        """
        return self._spec

    def attach_camera(
        self, name="camera", pos=[0, 0, 0], quat=[1, 0, 0, 0], fovy=45, mode="fixed"
    ):
        """
        Attaches a camera to the arena.

        Args:
            name: The name of the camera.
            pos: The position of the camera.
            quat: The orientation of the camera.
            fovy: The field of view of the camera.
            mode: The mode of the camera.

        Returns:
            The added camera.
        """
        camera = self._spec.worldbody.add_camera(
            name=name,
            fovy=fovy,
            pos=pos,
            quat=quat,
            # In MuJoCo's mjSpec, the mode is passed directly
            # If needed, this can be mapped to mj.mjtCamera enums
        )

        return camera
