import mujoco as mj


class StandardArena:
    def __init__(self) -> None:
        """
        Initializes the StandardArena object by creating a new MjSpec model and adding a solid color floor and lights.
        """
        self._arena = mj.MjSpec()

        # Set options
        self._arena.option.timestep = 0.002
        # TODO: set warm start

        _chequered = self._arena.add_texture(
            name="chequered",
            type=mj.mjtTexture.mjTEXTURE_2D,
            builtin=mj.mjtBuiltin.mjBUILTIN_CHECKER,
            width=300,
            height=300,
            rgb1=[0.2, 0.3, 0.4],
            rgb2=[0.3, 0.4, 0.5],
        )

        _grid = self._arena.add_material(
            name="grid", texrepeat=[5, 5], reflectance=0.2
        ).textures[mj.mjtTextureRole.mjTEXROLE_RGB] = "chequered"

        # add floor
        self._arena.worldbody.add_geom(
            type=mj.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1], material="grid"
        )

        # Add floor
        self._arena.worldbody.add_geom(
            type=mj.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1], material="grid"
        )

        # Add skybox
        self._arena.add_texture(
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
            self._arena.worldbody.add_light(pos=[x, -1, 3], dir=[-x, 1, -2])

        # TODO: check if this is needed
        # Enable copying during attachment to handle namespace conflicts
        # self._arena.copy_during_attach = True

        # count to see how many objects are added
        self._count = 0

    def attach(self, child_spec, pos=[0, 0, 0], quat=[1, 0, 0, 0]):
        """
        Attaches a child object to the arena at the specified position and orientation.
        """

        self._count += 1

        # Create a frame at the specified position and orientation
        _site = self._arena.worldbody.add_site(pos=pos, quat=quat)

        # Attach the child to this frame
        _body = _site.attach_body(child_spec.worldbody, "", "-" + str(self._count))

    def attach_free(self, child_spec, pos=[0, 0, 0], quat=[1, 0, 0, 0]):
        """
        Attaches a child object to the arena at the specified position and orientation with a free joint.
        """
        self._count += 1

        # Create a frame at the specified position and orientation
        _site = self._arena.worldbody.add_site(pos=pos, quat=quat)

        # Attach the child to this frame
        _body = _site.attach_body(child_spec.worldbody, "", "-" + str(self._count))

        _body.add_freejoint()

    @property
    def spec(self):
        """
        Returns the MjSpec for the StandardArena object.

        Returns:
            The MjSpec model.
        """
        return self._arena

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

        # TODO: verify if this is correct

        camera = self._arena.worldbody.add_camera(
            name=name,
            fovy=fovy,
            pos=pos,
            quat=quat,
        )

        return camera
