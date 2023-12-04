# Class that handles the construction of lattice boltzmann compute functions

import cupy as cp
import numba
import math
from numba import cuda

from xlb.compute_constructor.compute_constructor import ComputeConstructor


class NumbaNSE(NSE):
    """
    Class that handles the construction of lattice boltzmann compute functions using Numba

    Parameters
    ----------
    lattice : xlb.lattice.Lattice
        The lattice object that describes the lattice, e.g. D2Q9
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
        lattice,
        collision,
        stream,
        equilibrium,
        macroscopic,
        boundary_conditions=[],
        forcing=None,
        precision_policy=None,
    ):
        super().__init__(
            lattice,
            collision,
            stream,
            equilibrium,
            macroscopic,
            boundary_conditions,
            forcing,
            precision_policy,
        )

        # Construct numba kernels
        self._macroscopic_kernel = self._construct_macroscopic_kernel()
        self._step_kernel = self._construct_step_kernel()
        self._equilibrium_kernel = self._construct_equilibrium_kernel()

    def _construct_equilibrium_kernel(self):
        """
        Construct the equilibrium function kernel
        """

        # Get needed parameters for numba function
        d = self.lattice.d
        q = self.lattice.q
        equilibrium = self.lattice.construct_numba()

        # Make numba kernel
        @cuda.jit
        def _equilibrium(rho, u, feq):
            # Get the x and y indices
            ijk = cuda.grid(d)

            # Define local working arrays
            local_feq = cuda.local.array(q, numba.float32)

            # Get rho and u
            if d == 2:
                local_rho = rho[:, ijk[0], ijk[1]]
                local_u = u[:, ijk[0], ijk[1]]
            elif d == 3:
                local_rho = rho[:, ijk[0], ijk[1], ijk[2]]
                local_u = u[:, ijk[0], ijk[1], ijk[2]]

            # Compute equilibrium distribution
            local_feq = equilibrium(local_rho, local_u, local_feq)

            # Store equilibrium distribution
            for i in range(q):
                if d == 2:
                    feq[i, ijk[0], ijk[1]] = local_feq[i]
                elif d == 3:
                    feq[i, ijk[0], ijk[1], ijk[2]] = local_feq[i]

        return _equilibrium

    def equilibrium(self, rho, u, feq):
        """
        Compute the equilibrium distribution from the macroscopic variables
        """

        # Get shape of lattice
        nx, ny, nz = u.shape[1:]

        # Get block and grid sizes TODO: Make this a function
        threads_per_block = (8, 8, 8)
        blocks_per_grid_x = math.ceil(nx // threads_per_block[0])
        blocks_per_grid_y = math.ceil(ny // threads_per_block[1])
        blocks_per_grid_z = math.ceil(nz // threads_per_block[2])
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)

        # Perform step
        self._equilibrium[blocks_per_grid, threads_per_block](rho, u, feq)

    def macroscopic(self, f):
        """
        Compute the macroscopic variables from the distribution function

        Parameters
        ----------
        f : jax.numpy.ndarray
            The distribution function
        """
        raise NotImplementedError

    def _construct_step_kernel(self):
        """
        Construct the step function kernel
        """

        # Get needed parameters for numba function
        d = self.lattice.d
        q = self.lattice.q
        equilibrium = self.lattice.equilibrium_numba()
        streaming = self.lattice.streaming_numba()
        macroscopic = self.lattice.macroscopic_numba()
        collision = self.collision.collision_numba()

        # Make numba kernel
        @cuda.jit("void(float32[:,::1,::1,::1], float32[:,::1,::1,::1])")
        # @cuda.jit("void(float32[:,:,:,:], float32[:,:,:,:])")
        def _step(f0, f1):
            """
            Perform a single step of the lattice boltzmann method
            """

            # Get the x and y indices
            ijk = cuda.grid(d)

            # Define local working arrays
            f = cuda.local.array(q, numba.float32)
            rho = cuda.local.array(1, numba.float32)
            u = cuda.local.array(d, numba.float32)
            feq = cuda.local.array(q, numba.float32)
            fout = cuda.local.array(q, numba.float32)

            # Get f
            f = streaming(f0, f, ijk)

            # Compute macroscopic variables
            rho, u = macroscopic(f, rho, u)

            # Compute equilibrium distribution
            feq = equilibrium(rho, u, feq)

            # Perform collision
            fout = collision(f, feq, rho, u, fout)

            # Perform streaming
            for _ in range(q):
                if d == 2:
                    f1[ijk[0], ijk[1], _] = fout[_]
                elif d == 3:
                    f1[_, ijk[0], ijk[1], ijk[2]] = fout[_]

        return _step

    def step(self, f0, f1, timestep):
        """
        Perform a single step of the lattice boltzmann method
        """

        # Get shape of lattice
        nx, ny, nz = f0.shape[1:]

        # Get block and grid sizes
        threads_per_block = (8, 8, 8)
        blocks_per_grid_x = math.ceil(nx / threads_per_block[0])
        blocks_per_grid_y = math.ceil(ny / threads_per_block[1])
        blocks_per_grid_z = math.ceil(nz / threads_per_block[2])
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)

        # Perform step
        self._step[blocks_per_grid, threads_per_block](f0, f1)
