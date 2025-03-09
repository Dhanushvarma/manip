import mujoco as mj
import numpy as np


class Primitive:
    """
    A base class representing a primitive object in a simulation environment.
    """

    def __init__(self, **kwargs):
        """
        Initialize the Primitive object.

        Args:
            **kwargs: Additional keyword arguments for configuring the primitive.
        """
        self._spec = mj.MjSpec()

        # Handle type conversion if present (string to mjtGeom enum)
        if "type" in kwargs and isinstance(kwargs["type"], str):
            type_map = {
                "sphere": mj.mjtGeom.mjGEOM_SPHERE,
                "capsule": mj.mjtGeom.mjGEOM_CAPSULE,
                "cylinder": mj.mjtGeom.mjGEOM_CYLINDER,
                "box": mj.mjtGeom.mjGEOM_BOX,
                "plane": mj.mjtGeom.mjGEOM_PLANE,
                "mesh": mj.mjtGeom.mjGEOM_MESH,
                "ellipsoid": mj.mjtGeom.mjGEOM_ELLIPSOID,
            }
            if kwargs["type"] in type_map:
                kwargs["type"] = type_map[kwargs["type"]]

        # Add a geometric element to the worldbody
        self._geom = self._spec.worldbody.add_geom(**kwargs)

    @property
    def geom(self):
        """Returns the primitive's geom, e.g., to change color or friction."""
        return self._geom

    @property
    def spec(self):
        """Returns the primitive's mjSpec model."""
        return self._spec
