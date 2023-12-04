# Class that handles the construction of lattice boltzmann compute functions

import jax.numpy as jnp
from jax import jit
from functools import partial

from xlb.operator.boundary_condition.boundary_condition import ImplementationStep

class ComputeConstructor(object):
    """
    Base Class that handles the construction of all compute function in the lattice boltzmann method

    Parameters
    ----------
    velocity_set : xlb.velocity_set.VelocitySet
        The velocity set that describes the lattice
    collision : xlb.collision.Collision
        The collision object that describes the collision model, e.g. BGK
    boundary_conditions : list[xlb.boundary_conditions.BoundaryCondition]
        List of boundary conditions
    forcing : xlb.forcing.Forcing
        The forcing object that describes the forcing model. Defaults to None
    precision_policy : xlb.precision_policy.PrecisionPolicy
        The precision policy that describes the precision of the computation and store.
    """

    def __init__(
        self,
        velocity_set,
        collision,
        stream,
        equilibrium,
        macroscopic,
        boundary_conditions=[],
        forcing=None,
        precision_policy=None,
    ):
        # Set attributes
        self.velocity_set = velocity_set
        self.collision_op = collision
        self.stream_op = stream
        self.equilibrium_op = equilibrium
        self.macroscopic_op = macroscopic
        self.forcing_op = forcing
        self.precision_policy = precision_policy

        # Seperate Collision and Streaming boundary conditions
        self.boundary_conditions = boundary_conditions
        self.collision_boundary_conditions = []
        self.stream_boundary_conditions = []
        for bc in boundary_conditions:
            if bc.implementation_step == ImplementationStep.COLLISION:
                self.collision_boundary_conditions.append(bc)
            elif bc.implementation_step == ImplementationStep.STREAMING:
                self.stream_boundary_conditions.append(bc)
            else:
                raise ValueError("Boundary Condition has no implementation step")

    def generate_boundary_id(self):
        """
        Return the boundary id of a boundary condition

        Parameters
        ----------
        boundary : xlb.boundary_conditions.BoundaryCondition
            The boundary condition to get the id of
        """
        raise NotImplementedError("Must be implemented by subclass")

    def generate_mask(self):
        """
        Generate a mask that is True where the boundary condition is applied

        Parameters
        ----------
        boundary_id : int
            The boundary id of the boundary condition
        """
        raise NotImplementedError("Must be implemented by subclass")

    def equilibrium(self):
        """
        Compute the equilibrium distribution from the macroscopic variables
        """
        raise NotImplementedError("Must be implemented by subclass")

    def macroscopic(self):
        """
        Compute the macroscopic variables from the distribution function

        Parameters
        ----------
        f : jax.numpy.ndarray
            The distribution function
        """
        raise NotImplementedError("Must be implemented by subclass")

    def step(self):
        """
        Perform a single step of the lattice boltzmann method
        """
        raise NotImplementedError("Must be implemented by subclass")
