# Description: This file contains a simple example of using the OOCmap
# decorator to apply a function to a distributed array.
# Solves Lattice Boltzmann Taylor Green vortex decay

import mpi4py.MPI as MPI
import time
import warp as wp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import numpy as np
import cupy as cp
import time
from tqdm import tqdm

import phantomgaze as pg
import phantomgaze.render

try:
    import kvikio.nvcomp

    # GPU compressor lookup table
    gpu_compressor_lookup = {
        "cascaded": kvikio.nvcomp.CascadedManager,
        "Gdeflate": kvikio.nvcomp.GdeflateManager(),
        "snappy": kvikio.nvcomp.SnappyManager(),
        "LZ4": kvikio.nvcomp.LZ4Manager(),
        "bitcomp": kvikio.nvcomp.BitcompManager(),
    }
except ImportError:
    import warnings
    warnings.warn("kvikio not installed. Compression will not work.")
    gpu_compressor_lookup = {}

from out_of_core import OOCmap
from ooc_array import OOCArray

# Initialize MPI
comm = MPI.COMM_WORLD

# Disable pinned memory allocator
cp.cuda.set_pinned_memory_allocator(None)

# Initialize Warp
wp.init()


@wp.func
def int_to_float(x: wp.int32):
    """Convert a 32-bit int to a 32-bit float using math."""
    # Extract components
    sign = (x >> 31) & 1
    exp = ((x >> 23) & 0xFF) - 127
    fraction = x & ((1 << 23) - 1)
    
    # Compute the floating point value
    f = 1.0 + float(fraction) / float(1 << 23)
    f *= (2.0 ** float(exp))
    
    if sign:
        f = -f
    
    return f

@wp.func
def float_to_int(x: wp.float32):
    """Convert a 32-bit float to a 32-bit int using math."""
    # Handle special case for 0
    if x == 0.0:
        return 0
    
    # Extract sign bit
    if x < 0:
        sign = 1
        x = -x
    else:
        sign = 0
    
    # Extract exponent and fraction
    exp = wp.int(127)  # Bias for IEEE 754 32-bit float
    while x >= 2.0:
        x /= 2.0
        exp += 1
    while x < 1.0:
        x *= 2.0
        exp -= 1
    
    fraction = int((x - 1.0) * float((1 << 23)))
    
    # Construct the int using the sign, exponent, and fraction
    i = (sign << 31) | (exp << 23) | fraction

    return wp.int32(i)



@wp.func
def sample_f(
    f: wp.array4d(dtype=float),
    q: int,
    x: int,
    y: int,
    z: int,
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
    f_1_1_1 = sample_f(f0,  0,     x,     y,     z, width, height, length)
    f_2_1_1 = sample_f(f0,  1, x - 1,     y,     z, width, height, length)
    f_0_1_1 = sample_f(f0,  2, x + 1,     y,     z, width, height, length)
    f_1_2_1 = sample_f(f0,  3,     x, y - 1,     z, width, height, length)
    f_1_0_1 = sample_f(f0,  4,     x, y + 1,     z, width, height, length)
    f_1_1_2 = sample_f(f0,  5,     x,     y, z - 1, width, height, length)
    f_1_1_0 = sample_f(f0,  6,     x,     y, z + 1, width, height, length)
    f_1_2_2 = sample_f(f0,  7,     x, y - 1, z - 1, width, height, length)
    f_1_0_0 = sample_f(f0,  8,     x, y + 1, z + 1, width, height, length)
    f_1_2_0 = sample_f(f0,  9,     x, y - 1, z + 1, width, height, length)
    f_1_0_2 = sample_f(f0, 10,     x, y + 1, z - 1, width, height, length)
    f_2_1_2 = sample_f(f0, 11, x - 1,     y, z - 1, width, height, length)
    f_0_1_0 = sample_f(f0, 12, x + 1,     y, z + 1, width, height, length)
    f_2_1_0 = sample_f(f0, 13, x - 1,     y, z + 1, width, height, length)
    f_0_1_2 = sample_f(f0, 14, x + 1,     y, z - 1, width, height, length)
    f_2_2_1 = sample_f(f0, 15, x - 1, y - 1,     z, width, height, length)
    f_0_0_1 = sample_f(f0, 16, x + 1, y + 1,     z, width, height, length)
    f_2_0_1 = sample_f(f0, 17, x - 1, y + 1,     z, width, height, length)
    f_0_2_1 = sample_f(f0, 18, x + 1, y - 1,     z, width, height, length)

    # compute u and p
    p = (f_1_1_1
       + f_2_1_1 + f_0_1_1
       + f_1_2_1 + f_1_0_1
       + f_1_1_2 + f_1_1_0
       + f_1_2_2 + f_1_0_0
       + f_1_2_0 + f_1_0_2
       + f_2_1_2 + f_0_1_0
       + f_2_1_0 + f_0_1_2
       + f_2_2_1 + f_0_0_1
       + f_2_0_1 + f_0_2_1)
    u = (f_2_1_1 - f_0_1_1
       + f_2_1_2 - f_0_1_0
       + f_2_1_0 - f_0_1_2
       + f_2_2_1 - f_0_0_1
       + f_2_0_1 - f_0_2_1)
    v = (f_1_2_1 - f_1_0_1
       + f_1_2_2 - f_1_0_0
       + f_1_2_0 - f_1_0_2
       + f_2_2_1 - f_0_0_1
       - f_2_0_1 + f_0_2_1)
    w = (f_1_1_2 - f_1_1_0
       + f_1_2_2 - f_1_0_0
       - f_1_2_0 + f_1_0_2
       + f_2_1_2 - f_0_1_0
       - f_2_1_0 + f_0_1_2)
    res_p = 1.0 / p
    u = u * res_p
    v = v * res_p
    w = w * res_p
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
    factor_1 = 1.5
    factor_2 = 4.5
    weight_0 = 1.0 / 3.0
    weight_1 = 1.0 / 18.0
    weight_2 = 1.0 / 36.0
    f_eq_1_1_1 = weight_0 * (p * (factor_1 * (- uxu) + 1.0))
    f_eq_2_1_1 = weight_1 * (p * (factor_1 * (2.0 * exu_2_1_1 - uxu) + factor_2 * (exu_2_1_1 * exu_2_1_1) + 1.0))
    f_eq_0_1_1 = weight_1 * (p * (factor_1 * (2.0 * exu_0_1_1 - uxu) + factor_2 * (exu_0_1_1 * exu_0_1_1) + 1.0))
    f_eq_1_2_1 = weight_1 * (p * (factor_1 * (2.0 * exu_1_2_1 - uxu) + factor_2 * (exu_1_2_1 * exu_1_2_1) + 1.0))
    f_eq_1_0_1 = weight_1 * (p * (factor_1 * (2.0 * exu_1_0_1 - uxu) + factor_2 * (exu_1_2_1 * exu_1_2_1) + 1.0))
    f_eq_1_1_2 = weight_1 * (p * (factor_1 * (2.0 * exu_1_1_2 - uxu) + factor_2 * (exu_1_1_2 * exu_1_1_2) + 1.0))
    f_eq_1_1_0 = weight_1 * (p * (factor_1 * (2.0 * exu_1_1_0 - uxu) + factor_2 * (exu_1_1_0 * exu_1_1_0) + 1.0))
    f_eq_1_2_2 = weight_2 * (p * (factor_1 * (2.0 * exu_1_2_2 - uxu) + factor_2 * (exu_1_2_2 * exu_1_2_2) + 1.0))
    f_eq_1_0_0 = weight_2 * (p * (factor_1 * (2.0 * exu_1_0_0 - uxu) + factor_2 * (exu_1_0_0 * exu_1_0_0) + 1.0))
    f_eq_1_2_0 = weight_2 * (p * (factor_1 * (2.0 * exu_1_2_0 - uxu) + factor_2 * (exu_1_2_0 * exu_1_2_0) + 1.0))
    f_eq_1_0_2 = weight_2 * (p * (factor_1 * (2.0 * exu_1_0_2 - uxu) + factor_2 * (exu_1_0_2 * exu_1_0_2) + 1.0))
    f_eq_2_1_2 = weight_2 * (p * (factor_1 * (2.0 * exu_2_1_2 - uxu) + factor_2 * (exu_2_1_2 * exu_2_1_2) + 1.0))
    f_eq_0_1_0 = weight_2 * (p * (factor_1 * (2.0 * exu_0_1_0 - uxu) + factor_2 * (exu_0_1_0 * exu_0_1_0) + 1.0))
    f_eq_2_1_0 = weight_2 * (p * (factor_1 * (2.0 * exu_2_1_0 - uxu) + factor_2 * (exu_2_1_0 * exu_2_1_0) + 1.0))
    f_eq_0_1_2 = weight_2 * (p * (factor_1 * (2.0 * exu_0_1_2 - uxu) + factor_2 * (exu_0_1_2 * exu_0_1_2) + 1.0))
    f_eq_2_2_1 = weight_2 * (p * (factor_1 * (2.0 * exu_2_2_1 - uxu) + factor_2 * (exu_2_2_1 * exu_2_2_1) + 1.0))
    f_eq_0_0_1 = weight_2 * (p * (factor_1 * (2.0 * exu_0_0_1 - uxu) + factor_2 * (exu_0_0_1 * exu_0_0_1) + 1.0))
    f_eq_2_0_1 = weight_2 * (p * (factor_1 * (2.0 * exu_2_0_1 - uxu) + factor_2 * (exu_2_0_1 * exu_2_0_1) + 1.0))
    f_eq_0_2_1 = weight_2 * (p * (factor_1 * (2.0 * exu_0_2_1 - uxu) + factor_2 * (exu_0_2_1 * exu_0_2_1) + 1.0))

    # set next lattice state
    inv_tau = (1.0 / tau)
    f1[0, x, y, z] =  f_1_1_1 - inv_tau * (f_1_1_1 - f_eq_1_1_1)
    f1[1, x, y, z] =  f_2_1_1 - inv_tau * (f_2_1_1 - f_eq_2_1_1)
    f1[2, x, y, z] =  f_0_1_1 - inv_tau * (f_0_1_1 - f_eq_0_1_1)
    f1[3, x, y, z] =  f_1_2_1 - inv_tau * (f_1_2_1 - f_eq_1_2_1)
    f1[4, x, y, z] =  f_1_0_1 - inv_tau * (f_1_0_1 - f_eq_1_0_1)
    f1[5, x, y, z] =  f_1_1_2 - inv_tau * (f_1_1_2 - f_eq_1_1_2)
    f1[6, x, y, z] =  f_1_1_0 - inv_tau * (f_1_1_0 - f_eq_1_1_0)
    f1[7, x, y, z] =  f_1_2_2 - inv_tau * (f_1_2_2 - f_eq_1_2_2)
    f1[8, x, y, z] =  f_1_0_0 - inv_tau * (f_1_0_0 - f_eq_1_0_0)
    f1[9, x, y, z] =  f_1_2_0 - inv_tau * (f_1_2_0 - f_eq_1_2_0)
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
def f_to_eq_vp(
    f: wp.array4d(dtype=float),
    eq_vp: wp.array4d(dtype=wp.int32),
    width: int,
    height: int,
    length: int,
):

    # get index
    x, y, z = wp.tid()

    # sample needed points
    f_1_1_1 = sample_f(f,  0, x, y, z, width, height, length)
    f_2_1_1 = sample_f(f,  1, x, y, z, width, height, length)
    f_0_1_1 = sample_f(f,  2, x, y, z, width, height, length)
    f_1_2_1 = sample_f(f,  3, x, y, z, width, height, length)
    f_1_0_1 = sample_f(f,  4, x, y, z, width, height, length)
    f_1_1_2 = sample_f(f,  5, x, y, z, width, height, length)
    f_1_1_0 = sample_f(f,  6, x, y, z, width, height, length)
    f_1_2_2 = sample_f(f,  7, x, y, z, width, height, length)
    f_1_0_0 = sample_f(f,  8, x, y, z, width, height, length)
    f_1_2_0 = sample_f(f,  9, x, y, z, width, height, length)
    f_1_0_2 = sample_f(f, 10, x, y, z, width, height, length)
    f_2_1_2 = sample_f(f, 11, x, y, z, width, height, length)
    f_0_1_0 = sample_f(f, 12, x, y, z, width, height, length)
    f_2_1_0 = sample_f(f, 13, x, y, z, width, height, length)
    f_0_1_2 = sample_f(f, 14, x, y, z, width, height, length)
    f_2_2_1 = sample_f(f, 15, x, y, z, width, height, length)
    f_0_0_1 = sample_f(f, 16, x, y, z, width, height, length)
    f_2_0_1 = sample_f(f, 17, x, y, z, width, height, length)
    f_0_2_1 = sample_f(f, 18, x, y, z, width, height, length)

    # compute u and p
    p = (f_1_1_1
       + f_2_1_1 + f_0_1_1
       + f_1_2_1 + f_1_0_1
       + f_1_1_2 + f_1_1_0
       + f_1_2_2 + f_1_0_0
       + f_1_2_0 + f_1_0_2
       + f_2_1_2 + f_0_1_0
       + f_2_1_0 + f_0_1_2
       + f_2_2_1 + f_0_0_1
       + f_2_0_1 + f_0_2_1)
    u = (f_2_1_1 - f_0_1_1
       + f_2_1_2 - f_0_1_0
       + f_2_1_0 - f_0_1_2
       + f_2_2_1 - f_0_0_1
       + f_2_0_1 - f_0_2_1)
    v = (f_1_2_1 - f_1_0_1
       + f_1_2_2 - f_1_0_0
       + f_1_2_0 - f_1_0_2
       + f_2_2_1 - f_0_0_1
       - f_2_0_1 + f_0_2_1)
    w = (f_1_1_2 - f_1_1_0
       + f_1_2_2 - f_1_0_0
       - f_1_2_0 + f_1_0_2
       + f_2_1_2 - f_0_1_0
       - f_2_1_0 + f_0_1_2)
    res_p = 1.0 / p
    u = u * res_p
    v = v * res_p
    w = w * res_p
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
    weight_0 = 1.0 / 3.0
    weight_1 = 1.0 / 18.0
    weight_2 = 1.0 / 36.0
    f_eq_1_1_1 = weight_0 * (p * (factor_1 * (- uxu) + 1.0))
    f_eq_2_1_1 = weight_1 * (p * (factor_1 * (2.0 * exu_2_1_1 - uxu) + factor_2 * (exu_2_1_1 * exu_2_1_1) + 1.0))
    f_eq_0_1_1 = weight_1 * (p * (factor_1 * (2.0 * exu_0_1_1 - uxu) + factor_2 * (exu_0_1_1 * exu_0_1_1) + 1.0))
    f_eq_1_2_1 = weight_1 * (p * (factor_1 * (2.0 * exu_1_2_1 - uxu) + factor_2 * (exu_1_2_1 * exu_1_2_1) + 1.0))
    f_eq_1_0_1 = weight_1 * (p * (factor_1 * (2.0 * exu_1_0_1 - uxu) + factor_2 * (exu_1_2_1 * exu_1_2_1) + 1.0))
    f_eq_1_1_2 = weight_1 * (p * (factor_1 * (2.0 * exu_1_1_2 - uxu) + factor_2 * (exu_1_1_2 * exu_1_1_2) + 1.0))
    f_eq_1_1_0 = weight_1 * (p * (factor_1 * (2.0 * exu_1_1_0 - uxu) + factor_2 * (exu_1_1_0 * exu_1_1_0) + 1.0))
    f_eq_1_2_2 = weight_2 * (p * (factor_1 * (2.0 * exu_1_2_2 - uxu) + factor_2 * (exu_1_2_2 * exu_1_2_2) + 1.0))
    f_eq_1_0_0 = weight_2 * (p * (factor_1 * (2.0 * exu_1_0_0 - uxu) + factor_2 * (exu_1_0_0 * exu_1_0_0) + 1.0))
    f_eq_1_2_0 = weight_2 * (p * (factor_1 * (2.0 * exu_1_2_0 - uxu) + factor_2 * (exu_1_2_0 * exu_1_2_0) + 1.0))
    f_eq_1_0_2 = weight_2 * (p * (factor_1 * (2.0 * exu_1_0_2 - uxu) + factor_2 * (exu_1_0_2 * exu_1_0_2) + 1.0))
    f_eq_2_1_2 = weight_2 * (p * (factor_1 * (2.0 * exu_2_1_2 - uxu) + factor_2 * (exu_2_1_2 * exu_2_1_2) + 1.0))
    f_eq_0_1_0 = weight_2 * (p * (factor_1 * (2.0 * exu_0_1_0 - uxu) + factor_2 * (exu_0_1_0 * exu_0_1_0) + 1.0))
    f_eq_2_1_0 = weight_2 * (p * (factor_1 * (2.0 * exu_2_1_0 - uxu) + factor_2 * (exu_2_1_0 * exu_2_1_0) + 1.0))
    f_eq_0_1_2 = weight_2 * (p * (factor_1 * (2.0 * exu_0_1_2 - uxu) + factor_2 * (exu_0_1_2 * exu_0_1_2) + 1.0))
    f_eq_2_2_1 = weight_2 * (p * (factor_1 * (2.0 * exu_2_2_1 - uxu) + factor_2 * (exu_2_2_1 * exu_2_2_1) + 1.0))
    f_eq_0_0_1 = weight_2 * (p * (factor_1 * (2.0 * exu_0_0_1 - uxu) + factor_2 * (exu_0_0_1 * exu_0_0_1) + 1.0))
    f_eq_2_0_1 = weight_2 * (p * (factor_1 * (2.0 * exu_2_0_1 - uxu) + factor_2 * (exu_2_0_1 * exu_2_0_1) + 1.0))
    f_eq_0_2_1 = weight_2 * (p * (factor_1 * (2.0 * exu_0_2_1 - uxu) + factor_2 * (exu_0_2_1 * exu_0_2_1) + 1.0))

    # Store the equilibrium distribution in the f_eq array
    eq_vp[0, x, y, z] =  float_to_int(f_1_1_1) - float_to_int(f_eq_1_1_1)
    eq_vp[1, x, y, z] =  float_to_int(f_2_1_1) - float_to_int(f_eq_2_1_1)
    eq_vp[2, x, y, z] =  float_to_int(f_0_1_1) - float_to_int(f_eq_0_1_1)
    eq_vp[3, x, y, z] =  float_to_int(f_1_2_1) - float_to_int(f_eq_1_2_1)
    eq_vp[4, x, y, z] =  float_to_int(f_1_0_1) - float_to_int(f_eq_1_0_1)
    eq_vp[5, x, y, z] =  float_to_int(f_1_1_2) - float_to_int(f_eq_1_1_2)
    eq_vp[6, x, y, z] =  float_to_int(f_1_1_0) - float_to_int(f_eq_1_1_0)
    eq_vp[7, x, y, z] =  float_to_int(f_1_2_2) - float_to_int(f_eq_1_2_2)
    eq_vp[8, x, y, z] =  float_to_int(f_1_0_0) - float_to_int(f_eq_1_0_0)
    eq_vp[9, x, y, z] =  float_to_int(f_1_2_0) - float_to_int(f_eq_1_2_0)
    eq_vp[10, x, y, z] = float_to_int(f_1_0_2) - float_to_int(f_eq_1_0_2)
    eq_vp[11, x, y, z] = float_to_int(f_2_1_2) - float_to_int(f_eq_2_1_2)
    eq_vp[12, x, y, z] = float_to_int(f_0_1_0) - float_to_int(f_eq_0_1_0)
    eq_vp[13, x, y, z] = float_to_int(f_2_1_0) - float_to_int(f_eq_2_1_0)
    eq_vp[14, x, y, z] = float_to_int(f_0_1_2) - float_to_int(f_eq_0_1_2)
    eq_vp[15, x, y, z] = float_to_int(f_2_2_1) - float_to_int(f_eq_2_2_1)
    eq_vp[16, x, y, z] = float_to_int(f_0_0_1) - float_to_int(f_eq_0_0_1)
    eq_vp[17, x, y, z] = float_to_int(f_2_0_1) - float_to_int(f_eq_2_0_1)
    eq_vp[18, x, y, z] = float_to_int(f_0_2_1) - float_to_int(f_eq_0_2_1)
    eq_vp[19, x, y, z] = float_to_int(u)
    eq_vp[20, x, y, z] = float_to_int(v)
    eq_vp[21, x, y, z] = float_to_int(w)
    eq_vp[22, x, y, z] = float_to_int(p)


@wp.kernel
def eq_vp_to_f(
    eq_vp: wp.array4d(dtype=wp.int32),
    f: wp.array4d(dtype=float),
    width: int,
    height: int,
    length: int,
):

    # get index
    x, y, z = wp.tid()

    # sample needed points
    f_1_1_1 = eq_vp[ 0, x, y, z]
    f_2_1_1 = eq_vp[ 1, x, y, z]
    f_0_1_1 = eq_vp[ 2, x, y, z]
    f_1_2_1 = eq_vp[ 3, x, y, z]
    f_1_0_1 = eq_vp[ 4, x, y, z]
    f_1_1_2 = eq_vp[ 5, x, y, z]
    f_1_1_0 = eq_vp[ 6, x, y, z]
    f_1_2_2 = eq_vp[ 7, x, y, z]
    f_1_0_0 = eq_vp[ 8, x, y, z]
    f_1_2_0 = eq_vp[ 9, x, y, z]
    f_1_0_2 = eq_vp[10, x, y, z]
    f_2_1_2 = eq_vp[11, x, y, z]
    f_0_1_0 = eq_vp[12, x, y, z]
    f_2_1_0 = eq_vp[13, x, y, z]
    f_0_1_2 = eq_vp[14, x, y, z]
    f_2_2_1 = eq_vp[15, x, y, z]
    f_0_0_1 = eq_vp[16, x, y, z]
    f_2_0_1 = eq_vp[17, x, y, z]
    f_0_2_1 = eq_vp[18, x, y, z]
    u = int_to_float(eq_vp[19, x, y, z])
    v = int_to_float(eq_vp[20, x, y, z])
    w = int_to_float(eq_vp[21, x, y, z])
    p = int_to_float(eq_vp[22, x, y, z])
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
    weight_0 = 1.0 / 3.0
    weight_1 = 1.0 / 18.0
    weight_2 = 1.0 / 36.0
    f_eq_1_1_1 = weight_0 * (p * (factor_1 * (- uxu) + 1.0))
    f_eq_2_1_1 = weight_1 * (p * (factor_1 * (2.0 * exu_2_1_1 - uxu) + factor_2 * (exu_2_1_1 * exu_2_1_1) + 1.0))
    f_eq_0_1_1 = weight_1 * (p * (factor_1 * (2.0 * exu_0_1_1 - uxu) + factor_2 * (exu_0_1_1 * exu_0_1_1) + 1.0))
    f_eq_1_2_1 = weight_1 * (p * (factor_1 * (2.0 * exu_1_2_1 - uxu) + factor_2 * (exu_1_2_1 * exu_1_2_1) + 1.0))
    f_eq_1_0_1 = weight_1 * (p * (factor_1 * (2.0 * exu_1_0_1 - uxu) + factor_2 * (exu_1_2_1 * exu_1_2_1) + 1.0))
    f_eq_1_1_2 = weight_1 * (p * (factor_1 * (2.0 * exu_1_1_2 - uxu) + factor_2 * (exu_1_1_2 * exu_1_1_2) + 1.0))
    f_eq_1_1_0 = weight_1 * (p * (factor_1 * (2.0 * exu_1_1_0 - uxu) + factor_2 * (exu_1_1_0 * exu_1_1_0) + 1.0))
    f_eq_1_2_2 = weight_2 * (p * (factor_1 * (2.0 * exu_1_2_2 - uxu) + factor_2 * (exu_1_2_2 * exu_1_2_2) + 1.0))
    f_eq_1_0_0 = weight_2 * (p * (factor_1 * (2.0 * exu_1_0_0 - uxu) + factor_2 * (exu_1_0_0 * exu_1_0_0) + 1.0))
    f_eq_1_2_0 = weight_2 * (p * (factor_1 * (2.0 * exu_1_2_0 - uxu) + factor_2 * (exu_1_2_0 * exu_1_2_0) + 1.0))
    f_eq_1_0_2 = weight_2 * (p * (factor_1 * (2.0 * exu_1_0_2 - uxu) + factor_2 * (exu_1_0_2 * exu_1_0_2) + 1.0))
    f_eq_2_1_2 = weight_2 * (p * (factor_1 * (2.0 * exu_2_1_2 - uxu) + factor_2 * (exu_2_1_2 * exu_2_1_2) + 1.0))
    f_eq_0_1_0 = weight_2 * (p * (factor_1 * (2.0 * exu_0_1_0 - uxu) + factor_2 * (exu_0_1_0 * exu_0_1_0) + 1.0))
    f_eq_2_1_0 = weight_2 * (p * (factor_1 * (2.0 * exu_2_1_0 - uxu) + factor_2 * (exu_2_1_0 * exu_2_1_0) + 1.0))
    f_eq_0_1_2 = weight_2 * (p * (factor_1 * (2.0 * exu_0_1_2 - uxu) + factor_2 * (exu_0_1_2 * exu_0_1_2) + 1.0))
    f_eq_2_2_1 = weight_2 * (p * (factor_1 * (2.0 * exu_2_2_1 - uxu) + factor_2 * (exu_2_2_1 * exu_2_2_1) + 1.0))
    f_eq_0_0_1 = weight_2 * (p * (factor_1 * (2.0 * exu_0_0_1 - uxu) + factor_2 * (exu_0_0_1 * exu_0_0_1) + 1.0))
    f_eq_2_0_1 = weight_2 * (p * (factor_1 * (2.0 * exu_2_0_1 - uxu) + factor_2 * (exu_2_0_1 * exu_2_0_1) + 1.0))
    f_eq_0_2_1 = weight_2 * (p * (factor_1 * (2.0 * exu_0_2_1 - uxu) + factor_2 * (exu_0_2_1 * exu_0_2_1) + 1.0))

    # Store the equilibrium distribution in the f_eq array
    f[0, x, y, z] =  int_to_float(f_1_1_1 + float_to_int(f_eq_1_1_1))
    f[1, x, y, z] =  int_to_float(f_2_1_1 + float_to_int(f_eq_2_1_1))
    f[2, x, y, z] =  int_to_float(f_0_1_1 + float_to_int(f_eq_0_1_1))
    f[3, x, y, z] =  int_to_float(f_1_2_1 + float_to_int(f_eq_1_2_1))
    f[4, x, y, z] =  int_to_float(f_1_0_1 + float_to_int(f_eq_1_0_1))
    f[5, x, y, z] =  int_to_float(f_1_1_2 + float_to_int(f_eq_1_1_2))
    f[6, x, y, z] =  int_to_float(f_1_1_0 + float_to_int(f_eq_1_1_0))
    f[7, x, y, z] =  int_to_float(f_1_2_2 + float_to_int(f_eq_1_2_2))
    f[8, x, y, z] =  int_to_float(f_1_0_0 + float_to_int(f_eq_1_0_0))
    f[9, x, y, z] =  int_to_float(f_1_2_0 + float_to_int(f_eq_1_2_0))
    f[10, x, y, z] = int_to_float(f_1_0_2 + float_to_int(f_eq_1_0_2))
    f[11, x, y, z] = int_to_float(f_2_1_2 + float_to_int(f_eq_2_1_2))
    f[12, x, y, z] = int_to_float(f_0_1_0 + float_to_int(f_eq_0_1_0))
    f[13, x, y, z] = int_to_float(f_2_1_0 + float_to_int(f_eq_2_1_0))
    f[14, x, y, z] = int_to_float(f_0_1_2 + float_to_int(f_eq_0_1_2))
    f[15, x, y, z] = int_to_float(f_2_2_1 + float_to_int(f_eq_2_2_1))
    f[16, x, y, z] = int_to_float(f_0_0_1 + float_to_int(f_eq_0_0_1))
    f[17, x, y, z] = int_to_float(f_2_0_1 + float_to_int(f_eq_2_0_1))
    f[18, x, y, z] = int_to_float(f_0_2_1 + float_to_int(f_eq_0_2_1))


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
    weight_0 = 1.0 / 3.0
    weight_1 = 1.0 / 18.0
    weight_2 = 1.0 / 36.0
    f_eq_1_1_1 = weight_0 * (p * (factor_1 * (- uxu) + 1.0))
    f_eq_2_1_1 = weight_1 * (p * (factor_1 * (2.0 * exu_2_1_1 - uxu) + factor_2 * (exu_2_1_1 * exu_2_1_1) + 1.0))
    f_eq_0_1_1 = weight_1 * (p * (factor_1 * (2.0 * exu_0_1_1 - uxu) + factor_2 * (exu_0_1_1 * exu_0_1_1) + 1.0))
    f_eq_1_2_1 = weight_1 * (p * (factor_1 * (2.0 * exu_1_2_1 - uxu) + factor_2 * (exu_1_2_1 * exu_1_2_1) + 1.0))
    f_eq_1_0_1 = weight_1 * (p * (factor_1 * (2.0 * exu_1_0_1 - uxu) + factor_2 * (exu_1_2_1 * exu_1_2_1) + 1.0))
    f_eq_1_1_2 = weight_1 * (p * (factor_1 * (2.0 * exu_1_1_2 - uxu) + factor_2 * (exu_1_1_2 * exu_1_1_2) + 1.0))
    f_eq_1_1_0 = weight_1 * (p * (factor_1 * (2.0 * exu_1_1_0 - uxu) + factor_2 * (exu_1_1_0 * exu_1_1_0) + 1.0))
    f_eq_1_2_2 = weight_2 * (p * (factor_1 * (2.0 * exu_1_2_2 - uxu) + factor_2 * (exu_1_2_2 * exu_1_2_2) + 1.0))
    f_eq_1_0_0 = weight_2 * (p * (factor_1 * (2.0 * exu_1_0_0 - uxu) + factor_2 * (exu_1_0_0 * exu_1_0_0) + 1.0))
    f_eq_1_2_0 = weight_2 * (p * (factor_1 * (2.0 * exu_1_2_0 - uxu) + factor_2 * (exu_1_2_0 * exu_1_2_0) + 1.0))
    f_eq_1_0_2 = weight_2 * (p * (factor_1 * (2.0 * exu_1_0_2 - uxu) + factor_2 * (exu_1_0_2 * exu_1_0_2) + 1.0))
    f_eq_2_1_2 = weight_2 * (p * (factor_1 * (2.0 * exu_2_1_2 - uxu) + factor_2 * (exu_2_1_2 * exu_2_1_2) + 1.0))
    f_eq_0_1_0 = weight_2 * (p * (factor_1 * (2.0 * exu_0_1_0 - uxu) + factor_2 * (exu_0_1_0 * exu_0_1_0) + 1.0))
    f_eq_2_1_0 = weight_2 * (p * (factor_1 * (2.0 * exu_2_1_0 - uxu) + factor_2 * (exu_2_1_0 * exu_2_1_0) + 1.0))
    f_eq_0_1_2 = weight_2 * (p * (factor_1 * (2.0 * exu_0_1_2 - uxu) + factor_2 * (exu_0_1_2 * exu_0_1_2) + 1.0))
    f_eq_2_2_1 = weight_2 * (p * (factor_1 * (2.0 * exu_2_2_1 - uxu) + factor_2 * (exu_2_2_1 * exu_2_2_1) + 1.0))
    f_eq_0_0_1 = weight_2 * (p * (factor_1 * (2.0 * exu_0_0_1 - uxu) + factor_2 * (exu_0_0_1 * exu_0_0_1) + 1.0))
    f_eq_2_0_1 = weight_2 * (p * (factor_1 * (2.0 * exu_2_0_1 - uxu) + factor_2 * (exu_2_0_1 * exu_2_0_1) + 1.0))
    f_eq_0_2_1 = weight_2 * (p * (factor_1 * (2.0 * exu_0_2_1 - uxu) + factor_2 * (exu_0_2_1 * exu_0_2_1) + 1.0))

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
def initialize_f(eq_vp, f, dx: float):
    # Get inputs
    cs = 1.0 / np.sqrt(3.0)
    vel = 0.1 * cs
    (eq_vp, global_index) = eq_vp
    start_x, start_y, start_z = global_index[1], global_index[2], global_index[3]

    # Launch kernel to init f
    wp.launch(
        kernel=initialize_taylor_green,
        dim=list(f.shape[1:]),
        inputs=[f, dx, vel, start_x, start_y, start_z],
        device=f.device,
    )

    # Use f to set eq_vp
    wp.launch(
        kernel=f_to_eq_vp,
        dim=list(f.shape[1:]),
        inputs=[f, eq_vp, f.shape[1], f.shape[2], f.shape[3]],
        device=f.device,
    )

    return eq_vp


@OOCmap(comm, (0,), backend="warp")
def apply_stream_collide(eq_vp, f0, f1, tau: float, nr_steps: int):

    # Get f from eq_vp
    wp.launch(
        kernel=eq_vp_to_f,
        dim=list(f0.shape[1:]),
        inputs=[eq_vp, f0, f0.shape[1], f0.shape[2], f0.shape[3]],
        device=f0.device,
    )

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

    # Use f to set eq_vp
    wp.launch(
        kernel=f_to_eq_vp,
        dim=list(f0.shape[1:]),
        inputs=[f0, eq_vp, f0.shape[1], f0.shape[2], f0.shape[3]],
        device=f0.device,
    )

    return eq_vp


@OOCmap(comm, (), add_index=True, backend="cupy")
def render_f(eq_vp, img, depth, padding: int, camera, cmap):

    # Get global index
    (eq_vp, global_index) = eq_vp

    # Get norm v and u
    vel_p = eq_vp[19:, padding:-padding, padding:-padding, padding:-padding].view(dtype=cp.float32)
    norm_v = cp.linalg.norm(vel_p[:3], axis=0)
    u = vel_p[0, :, :, :]

    # Make volume for rendering
    u_volume = pg.Volume(
        u,
        spacing=(1.0, 1.0, 1.0),
        origin=(global_index[1]+padding, global_index[2]+padding, global_index[3]+padding),
    )
    norm_v_volume = pg.Volume(
        norm_v,
        spacing=(1.0, 1.0, 1.0),
        origin=(global_index[1]+padding+0.5, global_index[2]+padding+0.5, global_index[3]+padding+0.5),
    )

    # Render
    phantomgaze.render.contour(u_volume, camera, threshold=0.01, color=norm_v_volume, colormap=cmap, opacity=1.0, img=img, depth=depth)


if __name__ == "__main__":

    # Sim Parameters
    #n = 512
    n = 1024 + 256
    sub_n = 256
    tau = 0.505
    nr_sub_steps = 8
    dx = 2.0 * np.pi / n
    codec_name = "cascaded"

    # Make OOC distributed array
    eq_vp = OOCArray(
        shape=(23, n, n, n),
        dtype=np.int32,
        tile_shape=(23, sub_n, sub_n, sub_n),
        padding=(0, nr_sub_steps, nr_sub_steps, nr_sub_steps),
        comm=comm,
        devices=[cp.cuda.Device(0) for i in range(comm.size)],
        codec=gpu_compressor_lookup[codec_name] if codec_name in gpu_compressor_lookup else None,
        nr_compute_tiles=1,
    )

    # Make f0 and f1
    f0 = wp.empty(
        (
            19,
            sub_n + 2 * nr_sub_steps,
            sub_n + 2 * nr_sub_steps,
            sub_n + 2 * nr_sub_steps,
        ),
        dtype=wp.float32,
        device="cuda:0",
    )
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

    # Make camera
    img_res = (1080, 1920)
    camera = pg.Camera(
            position=(0.0, 0.0, -float(0.8*n)),
            focal_point=(n/2.0, n/2.0, n/2.0),
            view_up=(0.0, 1.0, 0.0),
            height=img_res[0],
            width=img_res[1],
    )
    cmap = pg.Colormap("jet", vmin=0.0, vmax=0.05)

    # Initialize f0
    eq_vp = initialize_f(eq_vp, f0, dx)

    # Apply streaming and collision
    nr_steps = 102400
    step = []
    comp_ratio = []
    t0 = time.time()
    for _ in tqdm(range(nr_steps)):

        # Store compression ratio
        step.append(_ * nr_sub_steps)
        comp_ratio.append(eq_vp.compression_ratio())
        print(f"Compression ratio: {eq_vp.compression_ratio()}")
        print(f"Size: {eq_vp.size() / 1024**3} GB")
        print("Nr Million Cells: ", n * n * n / 1e6)

        eq_vp = apply_stream_collide(eq_vp, f0, f1, tau, nr_sub_steps)

        if _ % 4 == 0:
            # Make image and depth
            img = cp.zeros((camera.height, camera.width, 4), dtype=cp.float32)
            img[:, :, 3] = 1.0
            depth = cp.ones((camera.height, camera.width), dtype=cp.float32) + np.nan

            # Render image
            render_f(eq_vp, img, depth, nr_sub_steps, camera, cmap)

            # Get image and save with matplotlib
            np_img = img.get()
            mpimg.imsave("img_" + str(_).zfill(5) + ".png", np_img)


    cp.cuda.Stream.null.synchronize()
    t1 = time.time()

    # Compute MLUPS
    mlups = (nr_sub_steps * nr_steps * n * n * n) / (t1 - t0) / 1e6
    print("Nr Million Cells: ", n * n * n / 1e6)
    print("MLUPS: ", mlups)

    # Plot results
    np_f = eq_vp.get_array()
    if comm.rank == 0:
        for i in range(23):
            plt.imshow(np_f[i, :, :, 0])
            plt.colorbar()
            plt.savefig("f_" + str(i) + ".png")
            plt.close()
