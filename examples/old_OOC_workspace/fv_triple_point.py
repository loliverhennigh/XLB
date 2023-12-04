# Description: This file contains a simple example of using the OOCmap
# decorator to apply a function to a distributed array.
# Solves Darcy flow in a 3D domain using a finite difference method.

import mpi4py.MPI as MPI
import time
import warp as wp
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cupy as cp
import time
from tqdm import tqdm
import kvikio.nvcomp
from tqdm import tqdm

from out_of_core import OOCmap
from ooc_array import OOCArray

# Initialize MPI
comm = MPI.COMM_WORLD

# Disable pinned memory allocator
cp.cuda.set_pinned_memory_allocator(None)

# Initialize Warp
wp.init()

# GPU compressor lookup table
gpu_compressor_lookup = {
    "cascaded": kvikio.nvcomp.CascadedManager(),
}

@wp.func
def index_state(
        state: wp.array3d(dtype=float),
        x: int,
        y: int,
        x_shape: int,
        y_shape: int):
    if x < 0:
        x = x_shape + x
    if y < 0:
        y = y_shape + y
    if x >= x_shape:
        x = x - x_shape
    if y >= y_shape:
        y = y - y_shape
    return wp.vec4(state[0, x, y], state[1, x, y], state[2, x, y], state[3, x, y])

@wp.func
def centeral_difference(
        v_0: wp.vec4,
        v_2: wp.vec4,
        dx: float):
    return (v_2 - v_0) / (2.0 * dx)

@wp.func
def slope_limiter_float(
        v_0: float,
        v_1: float,
        v_2: float,
        v_dx: float,
        dx: float):
    if v_dx == 0.0:
        dif_v_dx = v_dx + 1.0e-8
    v_dx = wp.max(
            0.0,
            wp.min(
                1.0, ((v_1 - v_0) / dx) / dif_v_dx
            ),
        ) * v_dx
    if v_dx == 0.0:
        dif_v_dx = v_dx + 1.0e-8
    v_dx = wp.max(
            0.0,
            wp.min(
                1.0, ((v_2 - v_1) / dx) / dif_v_dx
            ),
        ) * v_dx
    return v_dx

@wp.func
def slope_limiter(
        v_0: wp.vec4,
        v_1: wp.vec4,
        v_2: wp.vec4,
        v_dx: wp.vec4,
        dx: float):
    v_dx_0 = slope_limiter_float(v_0[0], v_1[0], v_2[0], v_dx[0], dx)
    v_dx_1 = slope_limiter_float(v_0[1], v_1[1], v_2[1], v_dx[1], dx)
    v_dx_2 = slope_limiter_float(v_0[2], v_1[2], v_2[2], v_dx[2], dx)
    v_dx_3 = slope_limiter_float(v_0[3], v_1[3], v_2[3], v_dx[3], dx)
    return wp.vec4(v_dx_0, v_dx_1, v_dx_2, v_dx_3)

@wp.func
def extrapolate_half_time_step(
        v: wp.vec4,
        v_dx: wp.vec4,
        v_dy: wp.vec4,
        dt: float,
        gamma: float):

    # Get values
    rho = v[0]
    rho_dx = v_dx[0]
    rho_dy = v_dy[0]
    vx = v[1]
    vx_dx = v_dx[1]
    vx_dy = v_dy[1]
    vy = v[2]
    vy_dx = v_dx[2]
    vy_dy = v_dy[2]
    p = v[3]
    p_dx = v_dx[3]
    p_dy = v_dy[3]

    # Extrapolate half time step
    rho_prime = rho - 0.5 * dt * (vx * rho_dx + rho * vx_dx + vy * rho_dy + rho * vy_dy)
    vx_prime = vx - 0.5 * dt * (vx * vx_dx + vy * vx_dy + (1.0 / rho) * p_dx)
    vy_prime = vy - 0.5 * dt * (vx * vy_dx + vy * vy_dy + (1.0 / rho) * p_dy)
    p_prime = p - 0.5 * dt * (gamma * p * (vx_dx + vy_dy) + vx * p_dx + vy * p_dy)

    return wp.vec4(rho_prime, vx_prime, vy_prime, p_prime)

@wp.func
def compute_flux(
        v_in: wp.vec4,
        v_out: wp.vec4,
        dim: int,
        sign: float,
        dx: float,
        dt: float,
        gamma: float):

    # Get values
    rho_in, rho_out = v_in[0], v_out[0]
    vx_in, vx_out = v_in[1], v_out[1]
    vy_in, vy_out = v_in[2], v_out[2]
    p_in, p_out = v_in[3], v_out[3]
    vdim_in, vdim_out = v_in[1+dim], v_out[1+dim] # direction of flux

    # Compute Energies
    e_in = p_in / (gamma - 1.0) + 0.5 * rho_in * (vx_in * vx_in + vy_in * vy_in)
    e_out = p_out / (gamma - 1.0) + 0.5 * rho_out * (vx_out * vx_out + vy_out * vy_out)

    # Compute averages
    average_rho = 0.5 * (rho_in + rho_out)
    average_momx = 0.5 * (rho_in * vx_in + rho_out * vx_out)
    average_momy = 0.5 * (rho_in * vy_in + rho_out * vy_out)
    average_e = 0.5 * (e_in + e_out)
    average_p = (gamma - 1.0) * (average_e - 0.5 * (average_momx * average_momx + average_momy * average_momy) / average_rho)
    if dim == 0:
        average_momdim = average_momx
    else:
        average_momdim = average_momy

    # Compute Wave Speeds
    c_in = wp.sqrt(gamma * p_in / rho_in) + wp.abs(vdim_in)
    c_out = wp.sqrt(gamma * p_out / rho_out) + wp.abs(vdim_out)
    c = wp.max(c_in, c_out)

    # Compute fluxes (Lax-Friedrichs)
    llf_flux_mass = average_momdim
    llf_flux_momx = average_momdim * average_momx / average_rho
    llf_flux_momy = average_momdim * average_momy / average_rho
    if dim == 0:
        llf_flux_momx += average_p
    else:
        llf_flux_momy += average_p
    llf_flux_energy = (average_e + average_p) * average_momdim / average_rho

    # Stabilizing diffusion term
    c_flux_mass = c * 0.5 * (rho_out - rho_in)
    c_flux_momx = c * 0.5 * (rho_out * vx_out - rho_in * vx_in)
    c_flux_momy = c * 0.5 * (rho_out * vy_out - rho_in * vy_in)
    c_flux_energy = c * 0.5 * (e_out - e_in)

    # Sum fluxes
    mass_flux = 1.0 * dt * dx * (sign * llf_flux_mass + c_flux_mass)
    momx_flux = 1.0 * dt * dx * (sign * llf_flux_momx + c_flux_momx)
    momy_flux = 1.0 * dt * dx * (sign * llf_flux_momy + c_flux_momy)
    energy_flux = 1.0 * dt * dx * (sign * llf_flux_energy + c_flux_energy)

    return wp.vec4(mass_flux, momx_flux, momy_flux, energy_flux)


@wp.kernel
def update(
        prim_state: wp.array3d(dtype=float),
        cons_state: wp.array3d(dtype=float),
        x_shape: int,
        y_shape: int,
        dx: float,
        dt: float,
        gamma: float):

    # Get index
    x, y = wp.tid()

    # Center Box
    v_1_1 = index_state(prim_state, x - 1, y - 1, x_shape, y_shape)
    v_1_2 = index_state(prim_state, x - 1, y, x_shape, y_shape)
    v_1_3 = index_state(prim_state, x - 1, y + 1, x_shape, y_shape)
    v_2_1 = index_state(prim_state, x, y - 1, x_shape, y_shape)
    v_2_2 = index_state(prim_state, x, y, x_shape, y_shape)
    v_2_3 = index_state(prim_state, x, y + 1, x_shape, y_shape)
    v_3_1 = index_state(prim_state, x + 1, y - 1, x_shape, y_shape)
    v_3_2 = index_state(prim_state, x + 1, y, x_shape, y_shape)
    v_3_3 = index_state(prim_state, x + 1, y + 1, x_shape, y_shape)

    # Star edges
    v_2_0 = index_state(prim_state, x, y - 2, x_shape, y_shape)
    v_2_4 = index_state(prim_state, x, y + 2, x_shape, y_shape)
    v_0_2 = index_state(prim_state, x - 2, y, x_shape, y_shape)
    v_4_2 = index_state(prim_state, x + 2, y, x_shape, y_shape)

    # Get derivatives
    # Center
    v_2_2_dx = centeral_difference(v_1_2, v_3_2, dx)
    v_2_2_dy = centeral_difference(v_2_1, v_2_3, dx)

    # Star
    v_2_1_dx = centeral_difference(v_1_1, v_3_1, dx)
    v_2_1_dy = centeral_difference(v_2_0, v_2_2, dx)
    v_2_3_dx = centeral_difference(v_1_3, v_3_3, dx)
    v_2_3_dy = centeral_difference(v_2_2, v_2_4, dx)
    v_1_2_dx = centeral_difference(v_0_2, v_2_2, dx)
    v_1_2_dy = centeral_difference(v_1_1, v_1_3, dx)
    v_3_2_dx = centeral_difference(v_2_2, v_4_2, dx)
    v_3_2_dy = centeral_difference(v_3_1, v_3_3, dx)

    # Apply slope limiter
    # Center
    v_2_2_dx = slope_limiter(v_1_2, v_2_2, v_3_2, v_2_2_dx, dx)
    v_2_2_dy = slope_limiter(v_2_1, v_2_2, v_2_3, v_2_2_dy, dx)

    # Star
    v_2_1_dx = slope_limiter(v_1_1, v_2_1, v_3_1, v_2_1_dx, dx)
    v_2_1_dy = slope_limiter(v_2_0, v_2_1, v_2_2, v_2_1_dy, dx)
    v_2_3_dx = slope_limiter(v_1_3, v_2_3, v_3_3, v_2_3_dx, dx)
    v_2_3_dy = slope_limiter(v_2_2, v_2_3, v_2_4, v_2_3_dy, dx)
    v_1_2_dx = slope_limiter(v_0_2, v_1_2, v_2_2, v_1_2_dx, dx)
    v_1_2_dy = slope_limiter(v_1_1, v_1_2, v_1_3, v_1_2_dy, dx)
    v_3_2_dx = slope_limiter(v_2_2, v_3_2, v_4_2, v_3_2_dx, dx)
    v_3_2_dy = slope_limiter(v_3_1, v_3_2, v_3_3, v_3_2_dy, dx)

    # Extrapolate half time step
    v_2_2_prime = extrapolate_half_time_step(v_2_2, v_2_2_dx, v_2_2_dy, dt, gamma)
    v_2_1_prime = extrapolate_half_time_step(v_2_1, v_2_1_dx, v_2_1_dy, dt, gamma)
    v_2_3_prime = extrapolate_half_time_step(v_2_3, v_2_3_dx, v_2_3_dy, dt, gamma)
    v_1_2_prime = extrapolate_half_time_step(v_1_2, v_1_2_dx, v_1_2_dy, dt, gamma)
    v_3_2_prime = extrapolate_half_time_step(v_3_2, v_3_2_dx, v_3_2_dy, dt, gamma)

    # Extrapolate in space to face
    v_2_0_face_in = v_2_2_prime - 0.5 * dx * v_2_2_dy
    v_2_0_face_out = v_2_1_prime + 0.5 * dx * v_2_1_dy
    v_2_4_face_in = v_2_2_prime + 0.5 * dx * v_2_2_dy
    v_2_4_face_out = v_2_3_prime - 0.5 * dx * v_2_3_dy
    v_0_2_face_in = v_2_2_prime - 0.5 * dx * v_2_2_dx
    v_0_2_face_out = v_1_2_prime + 0.5 * dx * v_1_2_dx
    v_4_2_face_in = v_2_2_prime + 0.5 * dx * v_2_2_dx
    v_4_2_face_out = v_3_2_prime - 0.5 * dx * v_3_2_dx

    # Get fluxes
    flux_2_0 = compute_flux(v_2_0_face_in, v_2_0_face_out, 1, 1.0, dx, dt, gamma)
    flux_2_4 = compute_flux(v_2_4_face_in, v_2_4_face_out, 1, -1.0, dx, dt, gamma)
    flux_0_2 = compute_flux(v_0_2_face_in, v_0_2_face_out, 0, 1.0, dx, dt, gamma)
    flux_4_2 = compute_flux(v_4_2_face_in, v_4_2_face_out, 0, -1.0, dx, dt, gamma)

    # Sum fluxes
    flux = flux_2_0 + flux_2_4 + flux_0_2 + flux_4_2

    # Apply fluxes
    cons_state[0, x, y] = cons_state[0, x, y] + flux[0]
    cons_state[1, x, y] = cons_state[1, x, y] + flux[1]
    cons_state[2, x, y] = cons_state[2, x, y] + flux[2]
    cons_state[3, x, y] = cons_state[3, x, y] + flux[3]

@wp.kernel
def get_conserved(
        prim_state: wp.array3d(dtype=float),
        cons_state: wp.array3d(dtype=float),
        gamma: float,
        vol: float):

    # get index
    i, j = wp.tid()

    # get primitive values
    rho = prim_state[0, i, j]
    vx = prim_state[1, i, j]
    vy = prim_state[2, i, j]
    p = prim_state[3, i, j]

    # get conserved values
    mass = rho * vol
    momx = rho * vx * vol
    momy = rho * vy * vol
    e = (p / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy)) * vol

    # set values
    cons_state[0, i, j] = mass
    cons_state[1, i, j] = momx
    cons_state[2, i, j] = momy
    cons_state[3, i, j] = e


@wp.kernel
def get_primitive(
        cons_state: wp.array3d(dtype=float),
        prim_state: wp.array3d(dtype=float),
        gamma: float,
        vol: float):

    # get index
    i, j = wp.tid()

    # get conserved values
    mass = cons_state[0, i, j]
    momx = cons_state[1, i, j]
    momy = cons_state[2, i, j]
    e = cons_state[3, i, j]

    # get primitive values
    rho = mass / vol
    vx = momx / rho / vol
    vy = momy / rho / vol
    p = (e / vol - 0.5 * rho * (vx * vx + vy * vy)) * (gamma - 1.0)

    # set values
    prim_state[0, i, j] = rho
    prim_state[1, i, j] = vx
    prim_state[2, i, j] = vy
    prim_state[3, i, j] = p


@wp.kernel
def initialize_kh(
        prim_state: wp.array3d(dtype=float),
        x_start: int,
        y_start: int,
        dx: float
):

    # Get spatial pos
    i,j = wp.tid()
    x = wp.float32(i + x_start) * dx
    y = wp.float32(j + y_start) * dx

    # Get primitive values
    if x < 0.5 and y < 0.5:
        rho = 0.138
        vx = 1.206
        vy = 1.206
        p = 0.029
    elif x >= 0.5 and y < 0.5:
        rho = 0.5323
        vx = 0.0
        vy = 1.206
        p = 0.3
    elif x < 0.5 and y >= 0.5:
        rho = 0.5323
        vx = 1.206
        vy = 0.0
        p = 0.3
    else:
        rho = 1.5
        vx = 0.0
        vy = 0.0
        p = 1.5

    # Set primitive values
    prim_state[0, i, j] = rho
    prim_state[1, i, j] = vx
    prim_state[2, i, j] = vy
    prim_state[3, i, j] = p


@OOCmap(comm, (0,), add_index=True, backend="warp")
def initialize_state(cons_state, prim_state, dx: float):
    # Get global index
    (cons_state, global_index) = cons_state
    start_x, start_y = global_index[1], global_index[2]

    # Launch kernel to initialize primitive state
    wp.launch(
        kernel=initialize_kh,
        dim=list(prim_state.shape[1:]),
        inputs=[prim_state, start_x, start_y, dx],
        device=prim_state.device,
    )

    # Launch kernel to get conserved state
    wp.launch(
        kernel=get_conserved,
        dim=list(cons_state.shape[1:]),
        inputs=[prim_state, cons_state, gamma, dx ** 2],
        device=cons_state.device,
    )

    return cons_state

@OOCmap(comm, (0,), add_index=True, backend="warp")
def fetch_rho(cons_state, np_rho, dx: float, padding: int):
    # Get global index
    (cons_state, global_index) = cons_state
    start_x, start_y = global_index[1], global_index[2]
    start_x += padding
    start_y += padding

    # Get numpy conserved state
    np_cons_state = cons_state.numpy()[0, padding:-padding, padding:-padding]
    np_rho[start_x:start_x + np_cons_state.shape[0], start_y:start_y + np_cons_state.shape[1]] = np_cons_state

    return cons_state

@OOCmap(comm, (0,), backend="warp")
def time_step(cons_state, prim_state, dx: float, gamma: float, dt: float, nr_steps: int):
    # run steps
    for i in range(nr_steps):
        # compute primitives
        wp.launch(get_primitive,
                  dim=prim_state.shape[1:],
                  inputs=[cons_state,
                          prim_state,
                          gamma, vol],
                  device=prim_state.device)

        # compute extrapolations to faces
        wp.launch(update,
                  dim=prim_state.shape[1:], 
                  inputs=[
                      prim_state,
                      cons_state,
                      prim_state.shape[1],
                      prim_state.shape[2],
                      dx,
                      dt,
                      gamma],
                  device=prim_state.device)

    return cons_state


if __name__ == "__main__":
    # simulation params
    scale_factor = 1
    res = 2048 * scale_factor
    gamma = 5.0 / 3.0
    courant_fac = 0.4
    dt = courant_fac * (1.0 / res) / (np.sqrt(gamma * 5.0) + 2.5) # hard set to smallest possible step needed
    shape = (res, res)
    dx = (1.0 / res)
    vol = (1.0 / res) ** 2
    tile_res = 512 * scale_factor
    tile_padding = 32 * scale_factor
    substeps = tile_padding - 2
    codec_name = "cascaded"
    codec = gpu_compressor_lookup[codec_name]
    steps = 1024 + 256
    plot_steps = 32
    print(f"Million cells: {res**2 / 1e6}")

    # allocate conservation quantities
    cons_state = OOCArray(
        shape=(4, res, res),
        dtype=np.float32,
        tile_shape=(4, tile_res, tile_res),
        padding=(0, tile_padding, tile_padding),
        comm=comm,
        devices=[cp.cuda.Device(0) for i in range(comm.size)],
        codec=codec,
    )

    # allocate primitive quantities
    prim_state = wp.empty(
            (4, tile_res + 2 * tile_padding, tile_res + 2 * tile_padding),
            dtype=wp.float32,
            device="cuda:0",
    )

    # allocate rho to be used for plotting
    np_rho = np.zeros((res, res), dtype=np.float32)

    # initialize fields
    cons_state = initialize_state(cons_state, prim_state, dx)

    # run steps
    t0 = time.time()
    for i in tqdm(range(steps)):
        # run step
        cons_state = time_step(cons_state, prim_state, dx, gamma, dt, substeps)

        # plot results
        if i % plot_steps == 0:
            # mass and compression ratio side by side
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            # plot Mass with colorbar
            fetch_rho(cons_state, np_rho, dx, tile_padding)
            axs[0].set_title("Density")
            fig.colorbar(axs[0].imshow(np_rho[::scale_factor, ::scale_factor], cmap='jet'), ax=axs[0])

            # Save rho
            np.save(f"triple_point.npy", np_rho)

            # plot compression ratio
            cp_ratio = np.zeros((res // tile_res, res // tile_res))
            for ii in range(res // tile_res):
                for jj in range(res // tile_res):
                    tile = cons_state.tiles[(0, ii, jj)]
                    cp_ratio[ii, jj] = tile.nbytes / tile.size()
            axs[1].set_title(f"Compression Ratio, Total: {cp_ratio.mean():.2f}")
            fig.colorbar(axs[1].imshow(cp_ratio, cmap="jet"), ax=axs[1])

            # save and close
            plt.savefig(f"triple_point_{str(i).zfill(4)}.png")
            plt.close()

    t1 = time.time()

    # Compute MLUPS
    mlups = (substeps * steps * res * res) / (t1 - t0) / 1e6
    print("Nr Million Cells: ", res * res / 1e6)
    print("MLUPS: ", mlups)

    ## Plot results
    #plt.imshow(np_cons_state[0, :, :])
    #plt.colorbar()
    #plt.savefig("cons.png")
    #plt.show()
