# Class that handles the construction of lattice boltzmann compute functions

import jax
import jax.numpy as jnp
from jax import jit
from jax import vmap
from functools import partial

from xlb.compute_constructor.nse.nse import NSE
from xlb.operator.stream import Stream
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.macroscopic import Macroscopic


class JaxNSE(NSE):
    """
    Class that handles the construction of lattice boltzmann compute functions using JAX

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
        stream=Stream(),
        equilibrium=QuadraticEquilibrium(),
        macroscopic=Macroscopic(),
        boundary_conditions=[],
        forcing=None,
        precision_policy=None,
    ):
        super().__init__(
            velocity_set,
            collision,
            stream,
            equilibrium,
            macroscopic,
            boundary_conditions,
            forcing,
            precision_policy,
        )

    @partial(jit, static_argnums=(0,), inline=True)
    def _apply_collision_boundary_conditions(self, f_pre, f_post, mask, collision_boundary_id):
        """
        Applies collision boundary conditions
        """

        # Construct branches for different boundary conditions
        # Do nothing, default branch
        @partial(jit)
        def _do_nothing(f_pre, f_post, mask):
            return f_post
        boundary_branch = [_do_nothing]

        # Make a branch for each boundary condition
        for bc in self.collision_boundary_conditions:
            @partial(jit)
            def _apply_jax(f_pre, f_post, mask):
                return bc.apply_jax(f_pre, f_post, mask, self.velocity_set)
            boundary_branch.append(_apply_jax)

        # vmap the boundary branch on all cells
        @partial(jnp.vectorize, signature='(n),(n),(n),()->(n)')
        def _boundary_switch(f_pre, f_post, mask, boundary_id):
            f = jax.lax.switch(
               boundary_id,
               boundary_branch,
               *(f_pre, f_post, mask)
            )
            return f

        return _boundary_switch(f_pre, f_post, mask, collision_boundary_id)

    @partial(jit, static_argnums=(0,), inline=True)
    def _apply_stream_boundary_conditions(self, f_pre, f_post, mask, stream_boundary_id):
        """
        Applies boundary conditions
        """

        # Construct branches for different boundary conditions
        # Do nothing, default branch
        @partial(jit)
        def _do_nothing(f_pre, f_post, mask):
            return f_post
        boundary_branch = [_do_nothing]

        # Make a branch for each boundary condition
        for bc in self.stream_boundary_conditions:
            @partial(jit)
            def _apply_jax(f_pre, f_post, mask):
                return bc.apply_jax(f_pre, f_post, mask, self.velocity_set)
            boundary_branch.append(_apply_jax)

        # vmap the boundary branch on all cells
        @partial(jnp.vectorize, signature='(n),(n),(n),()->(n)')
        def _boundary_switch(f_pre, f_post, mask, boundary_id):
            f = jax.lax.switch(
               boundary_id,
               boundary_branch,
               *(f_pre, f_post, mask)
            )
            return f

        return _boundary_switch(f_pre, f_post, mask, stream_boundary_id)



    @partial(jit, static_argnums=(0,), donate_argnums=(1, 2))
    def equilibrium(self, rho, u):
        """
        Compute the equilibrium distribution from the macroscopic variables
        """
        return self.equilibrium_op.apply_jax(
            rho,
            u,
            velocity_set=self.velocity_set,
        )

    @partial(jit, static_argnums=(0,), inline=True)
    def macroscopic(self, f):
        """
        Compute the macroscopic variables from the distribution function

        Parameters
        ----------
        f : jax.numpy.ndarray
            The distribution function
        """
        return self.macroscopic_op.apply_jax(
            f, velocity_set=self.velocity_set
        )

    @partial(jit, static_argnums=(0,), inline=True)
    def set_collision_boundary_id(self, boundary_id):
        """
        Construct the collision boundary id array
        """
        for id_, bc in enumerate(self.collision_boundary_conditions):
            boundary_id = bc.set_boundary_id(boundary_id, id_ + 1) # 0 is do nothing
        return boundary_id

    @partial(jit, static_argnums=(0,), inline=True)
    def set_stream_boundary_id(self, boundary_id):
        """
        Construct the stream boundary id array
        """
        for id_, bc in enumerate(self.stream_boundary_conditions):
            boundary_id = bc.set_boundary_id(boundary_id, id_ + 1)
        return boundary_id

    @partial(jit, static_argnums=(0,), inline=True)
    def set_mask(self, mask):
        """
        Sets the solid mask for the boundary condition.
        """
        # Set the solids in the mask
        for bc in self.boundary_conditions:
            mask = bc.set_solid_jax(mask)

        # Perform the streaming operation
        mask = self.stream_op.apply_jax(mask, self.velocity_set)

        return mask

    @partial(jit, static_argnums=(0, 5), donate_argnums=(1))
    def step(
            self,
            f,
            collision_boundary_id,
            stream_boundary_id,
            mask,
            timestep
        ):
        """
        Perform a single step of the lattice boltzmann method
        """

        # Cast to compute precision
        f_pre_collision = self.precision_policy.cast_to_compute_jax(f)

        # Compute the macroscopic variables
        rho, u = self.macroscopic(f_pre_collision)

        # Compute equilibrium
        feq = self.equilibrium(rho, u)

        # Apply collision
        f_post_collision = self.collision_op.apply_jax(
            f,
            feq,
            rho,
            u,
            velocity_set=self.velocity_set,
        )

        # Apply boundary conditions
        f_pre_streaming = self._apply_collision_boundary_conditions(
            f_pre_collision,
            f_post_collision,
            mask,
            collision_boundary_id,
        )

        ## Apply forcing
        #if self.forcing_op is not None:
        #    f = self.forcing_op.apply_jax(f, timestep)

        # Apply streaming
        f_post_streaming = self.stream_op.apply_jax(
            f_pre_streaming, self.velocity_set
        )

        # Apply boundary conditions
        f = self._apply_stream_boundary_conditions(
            f_pre_streaming,
            f_post_streaming,
            mask,
            stream_boundary_id,
        )

        # Copy back to store precision
        f = self.precision_policy.cast_to_store_jax(f)

        return f
