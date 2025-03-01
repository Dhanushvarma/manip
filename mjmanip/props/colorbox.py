import itertools
import numpy as np

from dm_control import manipulation
from dm_control import composer
from dm_control.manipulation.props.primitive import Box, _DEFAULT_HALF_LENGTHS


class ColorBoxWithSites(Box):
    """A class representing a box prop with sites on the corners."""

    def _build(self, half_lengths=None, mass=None, name="box", rgba=[1, 0, 0, 1.0]):
        half_lengths = half_lengths or _DEFAULT_HALF_LENGTHS
        super(ColorBoxWithSites, self)._build(
            half_lengths=half_lengths, mass=mass, name=name
        )

        self.geom.rgba = rgba
        print("Setting rgba to: ", rgba)

        corner_positions = itertools.product(
            [half_lengths[0], -half_lengths[0]],
            [half_lengths[1], -half_lengths[1]],
            [half_lengths[2], -half_lengths[2]],
        )
        corner_sites = []
        for i, corner_pos in enumerate(corner_positions):
            corner_sites.append(
                self._mjcf_root.worldbody.add(
                    "site",
                    type="sphere",
                    name="corner_{}".format(i),
                    size=[0.1],
                    pos=corner_pos,
                    rgba=rgba,
                    group=composer.SENSOR_SITES_GROUP,
                )
            )
        self._corner_sites = tuple(corner_sites)

    @property
    def corner_sites(self):
        return self._corner_sites
