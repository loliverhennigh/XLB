import jax

# jit compilation for q-criterion and vorticity
@jax.jit
def q_criterion(u):
    # Compute derivatives
    u_x = u[..., 0]
    u_y = u[..., 1]
    u_z = u[..., 2]

    # Compute derivatives
    u_x_dx = (u_x[2:, 1:-1, 1:-1] - u_x[:-2, 1:-1, 1:-1]) / 2
    u_x_dy = (u_x[1:-1, 2:, 1:-1] - u_x[1:-1, :-2, 1:-1]) / 2
    u_x_dz = (u_x[1:-1, 1:-1, 2:] - u_x[1:-1, 1:-1, :-2]) / 2
    u_y_dx = (u_y[2:, 1:-1, 1:-1] - u_y[:-2, 1:-1, 1:-1]) / 2
    u_y_dy = (u_y[1:-1, 2:, 1:-1] - u_y[1:-1, :-2, 1:-1]) / 2
    u_y_dz = (u_y[1:-1, 1:-1, 2:] - u_y[1:-1, 1:-1, :-2]) / 2
    u_z_dx = (u_z[2:, 1:-1, 1:-1] - u_z[:-2, 1:-1, 1:-1]) / 2
    u_z_dy = (u_z[1:-1, 2:, 1:-1] - u_z[1:-1, :-2, 1:-1]) / 2
    u_z_dz = (u_z[1:-1, 1:-1, 2:] - u_z[1:-1, 1:-1, :-2]) / 2

    # Compute vorticity
    mu_x = u_z_dy - u_y_dz
    mu_y = u_x_dz - u_z_dx
    mu_z = u_y_dx - u_x_dy
    norm_mu = jnp.sqrt(mu_x ** 2 + mu_y ** 2 + mu_z ** 2)

    # Compute strain rate
    s_0_0 = u_x_dx
    s_0_1 = 0.5 * (u_x_dy + u_y_dx)
    s_0_2 = 0.5 * (u_x_dz + u_z_dx)
    s_1_0 = s_0_1
    s_1_1 = u_y_dy
    s_1_2 = 0.5 * (u_y_dz + u_z_dy)
    s_2_0 = s_0_2
    s_2_1 = s_1_2
    s_2_2 = u_z_dz
    s_dot_s = (
        s_0_0 ** 2 + s_0_1 ** 2 + s_0_2 ** 2 +
        s_1_0 ** 2 + s_1_1 ** 2 + s_1_2 ** 2 +
        s_2_0 ** 2 + s_2_1 ** 2 + s_2_2 ** 2
    )

    # Compute omega
    omega_0_0 = 0.0
    omega_0_1 = 0.5 * (u_x_dy - u_y_dx)
    omega_0_2 = 0.5 * (u_x_dz - u_z_dx)
    omega_1_0 = -omega_0_1
    omega_1_1 = 0.0
    omega_1_2 = 0.5 * (u_y_dz - u_z_dy)
    omega_2_0 = -omega_0_2
    omega_2_1 = -omega_1_2
    omega_2_2 = 0.0
    omega_dot_omega = (
        omega_0_0 ** 2 + omega_0_1 ** 2 + omega_0_2 ** 2 +
        omega_1_0 ** 2 + omega_1_1 ** 2 + omega_1_2 ** 2 +
        omega_2_0 ** 2 + omega_2_1 ** 2 + omega_2_2 ** 2
    )

    # Compute q-criterion
    q = 0.5 * (omega_dot_omega - s_dot_s)

    return norm_mu, q


