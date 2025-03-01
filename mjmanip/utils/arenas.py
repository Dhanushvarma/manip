from dm_control.manipulation.shared.arenas import Standard


class CustomArena(Standard):

    def attach_offset(self, entity, offset, attach_site=None):
        """
        Attaches another entity at a position offset from the attachment site.

        Args:
        entity: The `Entity` to attach.
        offset: A length 7 array-like object representing the XYZ offset and WXYZ QUATERNION.
        attach_site: (optional) The site to which to attach the entity's model.
            If not set, defaults to self.attachment_site.
        Returns:
        The frame of the attached model.
        """
        frame = self.attach(entity, attach_site=attach_site)
        frame.pos = offset[:3]
        frame.quat = offset[3:]
        return frame
