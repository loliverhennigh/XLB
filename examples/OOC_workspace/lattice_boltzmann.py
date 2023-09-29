# Description: This file contains a simple example of using the OOCmap
# decorator to apply a function to a distributed array.
# Solves Darcy flow in a 3D domain using a finite difference method.

import mpi4py.MPI as MPI
import time
import warp as wp
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import time
from tqdm import tqdm

from out_of_core import OOCmap
from ooc_array import OOCArray

# Initialize MPI
comm = MPI.COMM_WORLD

# Initialize Warp
wp.init()


@wp.func
def sample(
    f: wp.array4d(dtype=float),
    x: int,
    y: int,
    z: int,
    q: int,
    width: int,
    height: int,
    length: int,
):

    # Periodic boundary conditions
    if x == -1:
        x = width - 1
    if x == width:
        x = 0
    if y == -1:
        y = height - 1
    if y == height:
        y = 0
    if z == -1:
        z = length - 1
    if z == length:
        z = 0
    s = f[q, x, y, z]
    return s


@wp.kernel
def stream_collide(
    f0: wp.array4d(dtype=float),
    f1: wp.array4d(dtype=float),
    width: int,
    height: int,
    length: int,
    tau: float,
):

    # get index
    x, y, z = wp.tid()

    # sample needed points
    f_1_1_1 = sample(f0, x, y, z, 0, width, height, length)
    f_2_1_1 = sample(f0, x, y, z - 1, 1, width, height, length)
    f_0_1_1 = sample(f0, x, y, z + 1, 2, width, height, length)
    f_1_2_1 = sample(f0, x, y - 1, z, 3, width, height, length)
    f_1_0_1 = sample(f0, x, y + 1, z, 4, width, height, length)
    f_1_1_2 = sample(f0, x - 1, y, z, 5, width, height, length)
    f_1_1_0 = sample(f0, x + 1, y, z, 6, width, height, length)
    f_1_2_2 = sample(f0, x - 1, y - 1, z, 7, width, height, length)
    f_1_0_0 = sample(f0, x + 1, y + 1, z, 8, width, height, length)
    f_1_2_0 = sample(f0, x + 1, y - 1, z, 9, width, height, length)
    f_1_0_2 = sample(f0, x - 1, y + 1, z, 10, width, height, length)
    f_2_1_2 = sample(f0, x - 1, y, z - 1, 11, width, height, length)
    f_0_1_0 = sample(f0, x + 1, y, z + 1, 12, width, height, length)
    f_2_1_0 = sample(f0, x + 1, y, z - 1, 13, width, height, length)
    f_0_1_2 = sample(f0, x - 1, y, z + 1, 14, width, height, length)
    f_2_2_1 = sample(f0, x, y - 1, z - 1, 15, width, height, length)
    f_0_0_1 = sample(f0, x, y + 1, z + 1, 16, width, height, length)
    f_2_0_1 = sample(f0, x, y + 1, z - 1, 17, width, height, length)
    f_0_2_1 = sample(f0, x, y - 1, z + 1, 18, width, height, length)

    # compute u and p
    p = (
        f_1_1_1
        + f_2_1_1
        + f_0_1_1
        + f_1_2_1
        + f_1_0_1
        + f_1_1_2
        + f_1_1_0
        + f_1_2_2
        + f_1_0_0
        + f_1_2_0
        + f_1_0_2
        + f_2_1_2
        + f_0_1_0
        + f_2_1_0
        + f_0_1_2
        + f_2_2_1
        + f_0_0_1
        + f_2_0_1
        + f_0_2_1
    )
    u = (
        f_2_1_1
        - f_0_1_1
        + f_2_1_2
        - f_0_1_0
        + f_2_1_0
        - f_0_1_2
        + f_2_2_1
        - f_0_0_1
        + f_2_0_1
        - f_0_2_1
    )
    v = (
        f_1_2_1
        - f_1_0_1
        + f_1_2_2
        - f_1_0_0
        + f_1_2_0
        - f_1_0_2
        + f_2_2_1
        - f_0_0_1
        - f_2_0_1
        + f_0_2_1
    )
    w = (
        f_1_1_2
        - f_1_1_0
        + f_1_2_2
        - f_1_0_0
        - f_1_2_0
        + f_1_0_2
        + f_2_1_2
        - f_0_1_0
        - f_2_1_0
        + f_0_1_2
    )
    u = u / p
    v = v / p
    w = w / p
    uxu = u * u + v * v + w * w

    # compute e dot u
    exu_1_1_1 = 0

    exu_2_1_1 = u
    exu_0_1_1 = -u

    exu_1_2_1 = v
    exu_1_0_1 = -v

    exu_1_1_2 = w
    exu_1_1_0 = -w

    exu_1_2_2 = v + w
    exu_1_0_0 = -v - w

    exu_1_2_0 = v - w
    exu_1_0_2 = -v + w

    exu_2_1_2 = u + w
    exu_0_1_0 = -u - w

    exu_2_1_0 = u - w
    exu_0_1_2 = -u + w

    exu_2_2_1 = u + v
    exu_0_0_1 = -u - v

    exu_2_0_1 = u - v
    exu_0_2_1 = -u + v

    # compute equilibrium dist
    denominator_1 = 0.6666666666
    denominator_2 = 0.1111111111
    weight_0 = 0.33333333
    weight_1 = 0.05555555
    weight_2 = 0.02777777
    f_eq_1_1_1 = weight_0 * (p * ((-uxu) / denominator_1 + 1.0))

    f_eq_2_1_1 = weight_1 * (
        p
        * (
            (2.0 * exu_2_1_1 - uxu) / denominator_1
            + 0.5 * ((exu_2_1_1 * exu_2_1_1) / denominator_2)
            + 1.0
        )
    )
    f_eq_0_1_1 = weight_1 * (
        p
        * (
            (2.0 * exu_0_1_1 - uxu) / denominator_1
            + 0.5 * ((exu_0_1_1 * exu_0_1_1) / denominator_2)
            + 1.0
        )
    )

    f_eq_1_2_1 = weight_1 * (
        p
        * (
            (2.0 * exu_1_2_1 - uxu) / denominator_1
            + 0.5 * ((exu_1_2_1 * exu_1_2_1) / denominator_2)
            + 1.0
        )
    )
    f_eq_1_0_1 = weight_1 * (
        p
        * (
            (2.0 * exu_1_0_1 - uxu) / denominator_1
            + 0.5 * ((exu_1_2_1 * exu_1_2_1) / denominator_2)
            + 1.0
        )
    )

    f_eq_1_1_2 = weight_1 * (
        p
        * (
            (2.0 * exu_1_1_2 - uxu) / denominator_1
            + 0.5 * ((exu_1_1_2 * exu_1_1_2) / denominator_2)
            + 1.0
        )
    )
    f_eq_1_1_0 = weight_1 * (
        p
        * (
            (2.0 * exu_1_1_0 - uxu) / denominator_1
            + 0.5 * ((exu_1_1_0 * exu_1_1_0) / denominator_2)
            + 1.0
        )
    )

    f_eq_1_2_2 = weight_2 * (
        p
        * (
            (2.0 * exu_1_2_2 - uxu) / denominator_1
            + 0.5 * ((exu_1_2_2 * exu_1_2_2) / denominator_2)
            + 1.0
        )
    )
    f_eq_1_0_0 = weight_2 * (
        p
        * (
            (2.0 * exu_1_0_0 - uxu) / denominator_1
            + 0.5 * ((exu_1_0_0 * exu_1_0_0) / denominator_2)
            + 1.0
        )
    )

    f_eq_1_2_0 = weight_2 * (
        p
        * (
            1.5 * (2.0 * exu_1_2_0 - uxu)
            + 0.5 * ((exu_1_2_0 * exu_1_2_0) / denominator_2)
            + 1.0
        )
    )
    f_eq_1_0_2 = weight_2 * (
        p
        * (
            (2.0 * exu_1_0_2 - uxu) / denominator_1
            + 0.5 * ((exu_1_0_2 * exu_1_0_2) / denominator_2)
            + 1.0
        )
    )

    f_eq_2_1_2 = weight_2 * (
        p
        * (
            (2.0 * exu_2_1_2 - uxu) / denominator_1
            + 0.5 * ((exu_2_1_2 * exu_2_1_2) / denominator_2)
            + 1.0
        )
    )
    f_eq_0_1_0 = weight_2 * (
        p
        * (
            (2.0 * exu_0_1_0 - uxu) / denominator_1
            + 0.5 * ((exu_0_1_0 * exu_0_1_0) / denominator_2)
            + 1.0
        )
    )

    f_eq_2_1_0 = weight_2 * (
        p
        * (
            (2.0 * exu_2_1_0 - uxu) / denominator_1
            + 0.5 * ((exu_2_1_0 * exu_2_1_0) / denominator_2)
            + 1.0
        )
    )
    f_eq_0_1_2 = weight_2 * (
        p
        * (
            (2.0 * exu_0_1_2 - uxu) / denominator_1
            + 0.5 * ((exu_0_1_2 * exu_0_1_2) / denominator_2)
            + 1.0
        )
    )

    f_eq_2_2_1 = weight_2 * (
        p
        * (
            (2.0 * exu_2_2_1 - uxu) / denominator_1
            + 0.5 * ((exu_2_2_1 * exu_2_2_1) / denominator_2)
            + 1.0
        )
    )
    f_eq_0_0_1 = weight_2 * (
        p
        * (
            (2.0 * exu_0_0_1 - uxu) / denominator_1
            + 0.5 * ((exu_0_0_1 * exu_0_0_1) / denominator_2)
            + 1.0
        )
    )

    f_eq_2_0_1 = weight_2 * (
        p
        * (
            (2.0 * exu_2_0_1 - uxu) / denominator_1
            + 0.5 * ((exu_2_0_1 * exu_2_0_1) / denominator_2)
            + 1.0
        )
    )
    f_eq_0_2_1 = weight_2 * (
        p
        * (
            (2.0 * exu_0_2_1 - uxu) / denominator_1
            + 0.5 * ((exu_0_2_1 * exu_0_2_1) / denominator_2)
            + 1.0
        )
    )

    # set next lattice state
    inv_tau = 1.0 / tau
    f1[0, x, y, z] = f_1_1_1 - inv_tau * (f_1_1_1 - f_eq_1_1_1)
    f1[1, x, y, z] = f_2_1_1 - inv_tau * (f_2_1_1 - f_eq_2_1_1)
    f1[2, x, y, z] = f_0_1_1 - inv_tau * (f_0_1_1 - f_eq_0_1_1)
    f1[3, x, y, z] = f_1_2_1 - inv_tau * (f_1_2_1 - f_eq_1_2_1)
    f1[4, x, y, z] = f_1_0_1 - inv_tau * (f_1_0_1 - f_eq_1_0_1)
    f1[5, x, y, z] = f_1_1_2 - inv_tau * (f_1_1_2 - f_eq_1_1_2)
    f1[6, x, y, z] = f_1_1_0 - inv_tau * (f_1_1_0 - f_eq_1_1_0)
    f1[7, x, y, z] = f_1_2_2 - inv_tau * (f_1_2_2 - f_eq_1_2_2)
    f1[8, x, y, z] = f_1_0_0 - inv_tau * (f_1_0_0 - f_eq_1_0_0)
    f1[9, x, y, z] = f_1_2_0 - inv_tau * (f_1_2_0 - f_eq_1_2_0)
    f1[10, x, y, z] = f_1_0_2 - inv_tau * (f_1_0_2 - f_eq_1_0_2)
    f1[11, x, y, z] = f_2_1_2 - inv_tau * (f_2_1_2 - f_eq_2_1_2)
    f1[12, x, y, z] = f_0_1_0 - inv_tau * (f_0_1_0 - f_eq_0_1_0)
    f1[13, x, y, z] = f_2_1_0 - inv_tau * (f_2_1_0 - f_eq_2_1_0)
    f1[14, x, y, z] = f_0_1_2 - inv_tau * (f_0_1_2 - f_eq_0_1_2)
    f1[15, x, y, z] = f_2_2_1 - inv_tau * (f_2_2_1 - f_eq_2_2_1)
    f1[16, x, y, z] = f_0_0_1 - inv_tau * (f_0_0_1 - f_eq_0_0_1)
    f1[17, x, y, z] = f_2_0_1 - inv_tau * (f_2_0_1 - f_eq_2_0_1)
    f1[18, x, y, z] = f_0_2_1 - inv_tau * (f_0_2_1 - f_eq_0_2_1)


@wp.kernel
def compute_vel_p(
    f0: wp.array4d(dtype=float),
    u0: wp.array4d(dtype=wp.vec3),
    p0: wp.array4d(dtype=float),
    width: int,
    height: int,
    length: int,
):

    # get index
    x, y, z = wp.tid()

    # sample needed points
    f_1_1_1 = sample(f0, x, y, z, 0, width, height, length)
    f_2_1_1 = sample(f0, x, y, z, 1, width, height, length)
    f_0_1_1 = sample(f0, x, y, z, 2, width, height, length)
    f_1_2_1 = sample(f0, x, y, z, 3, width, height, length)
    f_1_0_1 = sample(f0, x, y, z, 4, width, height, length)
    f_1_1_2 = sample(f0, x, y, z, 5, width, height, length)
    f_1_1_0 = sample(f0, x, y, z, 6, width, height, length)
    f_1_2_2 = sample(f0, x, y, z, 7, width, height, length)
    f_1_0_0 = sample(f0, x, y, z, 8, width, height, length)
    f_1_2_0 = sample(f0, x, y, z, 9, width, height, length)
    f_1_0_2 = sample(f0, x, y, z, 10, width, height, length)
    f_2_1_2 = sample(f0, x, y, z, 11, width, height, length)
    f_0_1_0 = sample(f0, x, y, z, 12, width, height, length)
    f_2_1_0 = sample(f0, x, y, z, 13, width, height, length)
    f_0_1_2 = sample(f0, x, y, z, 14, width, height, length)
    f_2_2_1 = sample(f0, x, y, z, 15, width, height, length)
    f_0_0_1 = sample(f0, x, y, z, 16, width, height, length)
    f_2_0_1 = sample(f0, x, y, z, 17, width, height, length)
    f_0_2_1 = sample(f0, x, y, z, 18, width, height, length)

    # compute u and p
    p = (
        f_1_1_1
        + f_2_1_1
        + f_0_1_1
        + f_1_2_1
        + f_1_0_1
        + f_1_1_2
        + f_1_1_0
        + f_1_2_2
        + f_1_0_0
        + f_1_2_0
        + f_1_0_2
        + f_2_1_2
        + f_0_1_0
        + f_2_1_0
        + f_0_1_2
        + f_2_2_1
        + f_0_0_1
        + f_2_0_1
        + f_0_2_1
    )
    u = (
        f_2_1_1
        - f_0_1_1
        + f_2_1_2
        - f_0_1_0
        + f_2_1_0
        - f_0_1_2
        + f_2_2_1
        - f_0_0_1
        + f_2_0_1
        - f_0_2_1
    )
    v = (
        f_1_2_1
        - f_1_0_1
        + f_1_2_2
        - f_1_0_0
        + f_1_2_0
        - f_1_0_2
        + f_2_2_1
        - f_0_0_1
        - f_2_0_1
        + f_0_2_1
    )
    w = (
        f_1_1_2
        - f_1_1_0
        + f_1_2_2
        - f_1_0_0
        - f_1_2_0
        + f_1_0_2
        + f_2_1_2
        - f_0_1_0
        - f_2_1_0
        + f_0_1_2
    )
    u = u / p
    v = v / p
    w = w / p

    # compute u and p
    p0[0, x, y, z] = p
    u0[0, x, y, z] = wp.vec3(u, v, w)


@wp.kernel
def initialize_taylor_green(
    f: wp.array4d(dtype=wp.float32),
    dx: float,
    vel: float,
    start_x: int,
    start_y: int,
    start_z: int,
):

    # get index
    i, j, k = wp.tid()

    # get real pos
    x = wp.float(i + start_x) * dx
    y = wp.float(j + start_y) * dx
    z = wp.float(k + start_z) * dx

    # compute u
    u = vel * wp.sin(x) * wp.cos(y) * wp.cos(z)
    v = -vel * wp.cos(x) * wp.sin(y) * wp.cos(z)
    w = 0.0

    # compute p
    p = (
        3.0
        * vel
        * vel
        * (1.0 / 16.0)
        * (wp.cos(2.0 * x) + wp.cos(2.0 * y) * (wp.cos(2.0 * z) + 2.0))
        + 1.0
    )

    # compute u X u
    uxu = u * u + v * v + w * w

    # compute e dot u
    exu_1_1_1 = 0.0
    exu_2_1_1 = u
    exu_0_1_1 = -u
    exu_1_2_1 = v
    exu_1_0_1 = -v
    exu_1_1_2 = w
    exu_1_1_0 = -w
    exu_1_2_2 = v + w
    exu_1_0_0 = -v - w
    exu_1_2_0 = v - w
    exu_1_0_2 = -v + w
    exu_2_1_2 = u + w
    exu_0_1_0 = -u - w
    exu_2_1_0 = u - w
    exu_0_1_2 = -u + w
    exu_2_2_1 = u + v
    exu_0_0_1 = -u - v
    exu_2_0_1 = u - v
    exu_0_2_1 = -u + v

    # compute equilibrium dist
    factor_1 = 1.5
    factor_2 = 4.5
    weight_0 = 0.33333333
    weight_1 = 0.05555555
    weight_2 = 0.02777777
    f_eq_1_1_1 = weight_0 * (p * (factor_1 * (-uxu) + 1.0))
    f_eq_2_1_1 = weight_1 * (
        p
        * (
            factor_1 * (2.0 * exu_2_1_1 - uxu)
            + factor_2 * (exu_2_1_1 * exu_2_1_1)
            + 1.0
        )
    )
    f_eq_0_1_1 = weight_1 * (
        p
        * (
            factor_1 * (2.0 * exu_0_1_1 - uxu)
            + factor_2 * (exu_0_1_1 * exu_0_1_1)
            + 1.0
        )
    )
    f_eq_1_2_1 = weight_1 * (
        p
        * (
            factor_1 * (2.0 * exu_1_2_1 - uxu)
            + factor_2 * (exu_1_2_1 * exu_1_2_1)
            + 1.0
        )
    )
    f_eq_1_0_1 = weight_1 * (
        p
        * (
            factor_1 * (2.0 * exu_1_0_1 - uxu)
            + factor_2 * (exu_1_2_1 * exu_1_2_1)
            + 1.0
        )
    )
    f_eq_1_1_2 = weight_1 * (
        p
        * (
            factor_1 * (2.0 * exu_1_1_2 - uxu)
            + factor_2 * (exu_1_1_2 * exu_1_1_2)
            + 1.0
        )
    )
    f_eq_1_1_0 = weight_1 * (
        p
        * (
            factor_1 * (2.0 * exu_1_1_0 - uxu)
            + factor_2 * (exu_1_1_0 * exu_1_1_0)
            + 1.0
        )
    )
    f_eq_1_2_2 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_1_2_2 - uxu)
            + factor_2 * (exu_1_2_2 * exu_1_2_2)
            + 1.0
        )
    )
    f_eq_1_0_0 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_1_0_0 - uxu)
            + factor_2 * (exu_1_0_0 * exu_1_0_0)
            + 1.0
        )
    )
    f_eq_1_2_0 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_1_2_0 - uxu)
            + factor_2 * (exu_1_2_0 * exu_1_2_0)
            + 1.0
        )
    )
    f_eq_1_0_2 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_1_0_2 - uxu)
            + factor_2 * (exu_1_0_2 * exu_1_0_2)
            + 1.0
        )
    )
    f_eq_2_1_2 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_2_1_2 - uxu)
            + factor_2 * (exu_2_1_2 * exu_2_1_2)
            + 1.0
        )
    )
    f_eq_0_1_0 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_0_1_0 - uxu)
            + factor_2 * (exu_0_1_0 * exu_0_1_0)
            + 1.0
        )
    )
    f_eq_2_1_0 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_2_1_0 - uxu)
            + factor_2 * (exu_2_1_0 * exu_2_1_0)
            + 1.0
        )
    )
    f_eq_0_1_2 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_0_1_2 - uxu)
            + factor_2 * (exu_0_1_2 * exu_0_1_2)
            + 1.0
        )
    )
    f_eq_2_2_1 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_2_2_1 - uxu)
            + factor_2 * (exu_2_2_1 * exu_2_2_1)
            + 1.0
        )
    )
    f_eq_0_0_1 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_0_0_1 - uxu)
            + factor_2 * (exu_0_0_1 * exu_0_0_1)
            + 1.0
        )
    )
    f_eq_2_0_1 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_2_0_1 - uxu)
            + factor_2 * (exu_2_0_1 * exu_2_0_1)
            + 1.0
        )
    )
    f_eq_0_2_1 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_0_2_1 - uxu)
            + factor_2 * (exu_0_2_1 * exu_0_2_1)
            + 1.0
        )
    )

    # set next lattice state
    f[0, i, j, k] = f_eq_1_1_1
    f[1, i, j, k] = f_eq_2_1_1
    f[2, i, j, k] = f_eq_0_1_1
    f[3, i, j, k] = f_eq_1_2_1
    f[4, i, j, k] = f_eq_1_0_1
    f[5, i, j, k] = f_eq_1_1_2
    f[6, i, j, k] = f_eq_1_1_0
    f[7, i, j, k] = f_eq_1_2_2
    f[8, i, j, k] = f_eq_1_0_0
    f[9, i, j, k] = f_eq_1_2_0
    f[10, i, j, k] = f_eq_1_0_2
    f[11, i, j, k] = f_eq_2_1_2
    f[12, i, j, k] = f_eq_0_1_0
    f[13, i, j, k] = f_eq_2_1_0
    f[14, i, j, k] = f_eq_0_1_2
    f[15, i, j, k] = f_eq_2_2_1
    f[16, i, j, k] = f_eq_0_0_1
    f[17, i, j, k] = f_eq_2_0_1
    f[18, i, j, k] = f_eq_0_2_1


@OOCmap(comm, (0,), add_index=True, backend="warp")
def initialize_f(f, dx: float):
    # Get inputs
    cs = 1.0 / np.sqrt(3.0)
    vel = 0.1 * cs
    (f, global_index) = f
    start_x, start_y, start_z = global_index[1], global_index[2], global_index[3]

    # Launch kernel
    wp.launch(
        kernel=initialize_taylor_green,
        dim=list(f.shape[1:]),
        inputs=[f, dx, vel, start_x, start_y, start_z],
        device=f.device,
    )

    return f


@OOCmap(comm, (0,), backend="warp")
def apply_stream_collide(f0, f1, tau: float, nr_steps: int):
    # f0: is assumed to be a OOC array

    # Apply streaming and collision for nr_steps
    for _ in range(nr_steps):
        # Apply streaming and collision step
        wp.launch(
            kernel=stream_collide,
            dim=list(f0.shape[1:]),
            inputs=[f0, f1, f0.shape[1], f0.shape[2], f0.shape[3], tau],
            device=f0.device,
        )

        # Swap f0 and f1
        f0, f1 = f1, f0

    return f0


if __name__ == "__main__":

    # Sim Parameters
    n = 600
    sub_n = 300
    tau = 0.53
    nr_sub_steps = 16
    dx = 2.0 * np.pi / n

    # Make OOC distributed array
    f0 = OOCArray(
        shape=(19, n, n, n),
        dtype=np.float32,
        tile_shape=(19, sub_n, sub_n, sub_n),
        padding=(0, nr_sub_steps, nr_sub_steps, nr_sub_steps),
        comm=comm,
        devices=["cuda:0"],
    )

    # Make f1
    f1 = wp.empty(
        (
            19,
            sub_n + 2 * nr_sub_steps,
            sub_n + 2 * nr_sub_steps,
            sub_n + 2 * nr_sub_steps,
        ),
        dtype=wp.float32,
        device="cuda:0",
    )

    # Initialize f0
    f0 = initialize_f(f0, dx)

    # Apply streaming and collision
    nr_steps = 8
    t0 = time.time()
    for _ in tqdm(range(nr_steps)):
        f0 = apply_stream_collide(f0, f1, tau, nr_sub_steps)

        #if _ % 1 == 0:
        #    # Plot results
        #    np_f = f0.get_array()
        #    plt.imshow(np_f[3, :, :, 200])
        #    plt.colorbar()
        #    plt.savefig("f_.png")
        #    plt.show()
    t1 = time.time()

    # Compute MLUPS
    mlups = (nr_sub_steps * nr_steps * n * n * n) / (t1 - t0) / 1e6
    print("Nr Million Cells: ", n * n * n / 1e6)
    print("MLUPS: ", mlups)

    # Plot results
    np_f = f0.get_array()
    plt.imshow(np_f[3, :, :, 200])
    plt.colorbar()
    plt.savefig("f_.png")
    plt.show()
