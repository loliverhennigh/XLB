# Base Lattice class

import re
import numpy as np
import jax.numpy as jnp
from jax import jit, lax, vmap
from functools import partial

from xlb.precision_policy import PrecisionPolicy


class Lattice(object):
    """
    Base class for lattice.

    Parameters
    ----------
    d: int
        The dimension of the lattice.
    q: int
        The number of velocities of the lattice.
    precision_policy: PrecisionPolicy
        The precision policy for the lattice.
    """

    def __init__(
        self, d, q, precision_policy=PrecisionPolicy(jnp.float32, jnp.float32)
    ):
        # Store the dimension and the number of velocities
        self.d = d
        self.q = q
        self.precision_policy = precision_policy

        # Construct the properties of the lattice
        self.c = self._construct_lattice_velocity()
        self.w = self._construct_lattice_weight()
        self.cc = self._construct_lattice_moment()
        self.opp_indices = self._construct_opposite_indices()
        self.main_indices = self._construct_main_indices()
        self.right_indices = self._construct_right_indices()
        self.left_indices = self._construct_left_indices()

    def construct_lattice_velocity(self):
        """
        This function constructs the velocity vectors of the lattice.

        The velocity vectors are defined based on the name of the lattice. For example, for a D2Q9
        lattice, there are 9 velocities: (0,0), (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1),
        (1,-1), and (-1,1).

        Returns
        -------
        c.T: numpy.ndarray
            The velocity vectors of the lattice.
        """
        if self.name == "D2Q9":  # D2Q9
            cx = [0, 0, 0, 1, -1, 1, -1, 1, -1]
            cy = [0, 1, -1, 0, 1, -1, 0, 1, -1]
            c = np.array(tuple(zip(cx, cy)))
        elif self.name == "D3Q19":  # D3Q19
            c = [(x, y, z) for x in [0, -1, 1] for y in [0, -1, 1] for z in [0, -1, 1]]
            c = np.array([ci for ci in c if np.linalg.norm(ci) < 1.5])
        elif self.name == "D3Q27":  # D3Q27
            c = [(x, y, z) for x in [0, -1, 1] for y in [0, -1, 1] for z in [0, -1, 1]]
            # c = np.array([ci for ci in c if np.linalg.norm(ci) < 1.5])
            c = np.array(c)
        else:
            raise ValueError("Supported Lattice types are D2Q9, D3Q19 and D3Q27")

        return c.T

    def _construct_opposite_indices(self):
        """
        This function constructs the indices of the opposite velocities for each velocity.

        The opposite velocity of a velocity is the velocity that has the same magnitude but the
        opposite direction.

        Returns
        -------
        opposite: numpy.ndarray
            The indices of the opposite velocities.
        """
        c = self.c.T
        opposite = np.array([c.tolist().index((-c[i]).tolist()) for i in range(self.q)])
        return opposite

    def _construct_right_indices(self):
        """
        This function constructs the indices of the velocities that point in the positive
        x-direction.

        Returns
        -------
        numpy.ndarray
            The indices of the right velocities.
        """
        c = self.c.T
        return np.nonzero(c[:, 0] == 1)[0]

    def construct_left_indices(self):
        """
        This function constructs the indices of the velocities that point in the negative
        x-direction.

        Returns
        -------
        numpy.ndarray
            The indices of the left velocities.
        """
        c = self.c.T
        return np.nonzero(c[:, 0] == -1)[0]

    def construct_main_indices(self):
        """
        This function constructs the indices of the main velocities.

        The main velocities are the velocities that have a magnitude of 1 in lattice units.

        Returns
        -------
        numpy.ndarray
            The indices of the main velocities.
        """
        c = self.c.T
        if self.d == 2:
            return np.nonzero((np.abs(c[:, 0]) + np.abs(c[:, 1]) == 1))[0]

        elif self.d == 3:
            return np.nonzero(
                (np.abs(c[:, 0]) + np.abs(c[:, 1]) + np.abs(c[:, 2]) == 1)
            )[0]

    def construct_lattice_velocity(self):
        """
        This function constructs the velocity vectors of the lattice.

        The velocity vectors are defined based on the name of the lattice. For example, for a D2Q9
        lattice, there are 9 velocities: (0,0), (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1),
        (1,-1), and (-1,1).

        Returns
        -------
        c.T: numpy.ndarray
            The velocity vectors of the lattice.
        """
        if self.name == "D2Q9":  # D2Q9
            cx = [0, 0, 0, 1, -1, 1, -1, 1, -1]
            cy = [0, 1, -1, 0, 1, -1, 0, 1, -1]
            c = np.array(tuple(zip(cx, cy)))
        elif self.name == "D3Q19":  # D3Q19
            c = [(x, y, z) for x in [0, -1, 1] for y in [0, -1, 1] for z in [0, -1, 1]]
            c = np.array([ci for ci in c if np.linalg.norm(ci) < 1.5])
        elif self.name == "D3Q27":  # D3Q27
            c = [(x, y, z) for x in [0, -1, 1] for y in [0, -1, 1] for z in [0, -1, 1]]
            # c = np.array([ci for ci in c if np.linalg.norm(ci) < 1.5])
            c = np.array(c)
        else:
            raise ValueError("Supported Lattice types are D2Q9, D3Q19 and D3Q27")

        return c.T

    def construct_lattice_weight(self):
        """
        This function constructs the weights of the lattice.

        The weights are defined based on the name of the lattice. For example, for a D2Q9 lattice,
        the weights are 4/9 for the rest velocity, 1/9 for the main velocities, and 1/36 for the
        diagonal velocities.

        Returns
        -------
        w: numpy.ndarray
            The weights of the lattice.
        """
        # Get the transpose of the lattice vector
        c = self.c.T

        # Initialize the weights to be 1/36
        w = 1.0 / 36.0 * np.ones(self.q)

        # Update the weights for 2D and 3D lattices
        if self.name == "D2Q9":
            w[np.linalg.norm(c, axis=1) < 1.1] = 1.0 / 9.0
            w[0] = 4.0 / 9.0
        elif self.name == "D3Q19":
            w[np.linalg.norm(c, axis=1) < 1.1] = 2.0 / 36.0
            w[0] = 1.0 / 3.0
        elif self.name == "D3Q27":
            cl = np.linalg.norm(c, axis=1)
            w[np.isclose(cl, 1.0, atol=1e-8)] = 2.0 / 27.0
            w[(cl > 1) & (cl <= np.sqrt(2))] = 1.0 / 54.0
            w[(cl > np.sqrt(2)) & (cl <= np.sqrt(3))] = 1.0 / 216.0
            w[0] = 8.0 / 27.0
        else:
            raise ValueError("Supported Lattice types are D2Q9, D3Q19 and D3Q27")

        # Return the weights
        return w

    def construct_lattice_moment(self):
        """
        This function constructs the moments of the lattice.

        The moments are the products of the velocity vectors, which are used in the computation of
        the equilibrium distribution functions and the collision operator in the Lattice Boltzmann
        Method (LBM).

        Returns
        -------
        cc: numpy.ndarray
            The moments of the lattice.
        """
        c = self.c.T
        # Counter for the loop
        cntr = 0

        # nt: number of independent elements of a symmetric tensor
        nt = self.d * (self.d + 1) // 2

        cc = np.zeros((self.q, nt))
        for a in range(0, self.d):
            for b in range(a, self.d):
                cc[:, cntr] = c[:, a] * c[:, b]
                cntr += 1

        return cc

    def __str__(self):
        return "D{}Q{}".format(self.d, self.q)
