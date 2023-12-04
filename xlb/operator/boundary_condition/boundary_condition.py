"""
Base class for boundary conditions in a LBM simulation.
"""

import jax.numpy as jnp
from jax import jit, device_count
from functools import partial
import numpy as np
from enum import Enum

from xlb.operator.operator import Operator
from xlb.velocity_set.velocity_set import VelocitySet

# Enum for implementation step
class ImplementationStep(Enum):
    COLLISION = 1
    STREAMING = 2

class BoundaryCondition(Operator):
    """
    Base class for boundary conditions in a LBM simulation.
    """

    def __init__(
            self,
            set_mask,
            implementation_step: ImplementationStep
        ):
        self._set_mask = set_mask
        self.implementation_step = implementation_step

    @classmethod
    def from_indices(cls, indices, implementation_step: ImplementationStep):
        """
        Creates a boundary condition from a list of indices.
        """
        raise NotImplementedError

    @partial(jit, static_argnums=(0), donate_argnums=(1))
    def set_mask(self, mask_id, id_number):
        """
        Sets the mask id for the boundary condition.
        """
        return self._set_mask(boundary_id, mask, id_number)

    @partial(jit, static_argnums=(0,))
    def apply_jax(self, f_pre, f_post, mask, velocity_set: VelocitySet):
        """
        Applies the boundary condition.
        """
        pass
