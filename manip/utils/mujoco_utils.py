import mujoco as mj
import numpy as np


def get_site_jac(model, data, site_id):
    """Return the Jacobian (translational and rotational components) of the
    end-effector of the corresponding site id.
    """
    # Create arrays for the Jacobians
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    # Call the Jacobian calculation function
    mj.mj_jacSite(model, data, jacp, jacr, site_id)

    # Stack the translational and rotational Jacobians
    jac = np.vstack([jacp, jacr])
    return jac


def get_fullM(model, data):
    """Compute full mass matrix from factorized mass matrix.

    Args:
        model: MjModel instance
        data: MjData instance

    Returns:
        M: Full mass matrix (nv x nv)
    """
    # Initialize matrix for the result
    M = np.zeros((model.nv, model.nv))

    # Call the function to compute the full mass matrix
    mj.mj_fullM(model, M, data.qM)

    return M
