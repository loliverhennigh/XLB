"""
This is a example for simulating fluid flow around a NACA airfoil using the lattice Boltzmann method (LBM). 
The LBM is a computational fluid dynamics method for simulating fluid flow and is particularly effective 
for complex geometries and multiphase flow. 

In this example you'll be introduced to the following concepts:

1. Lattice: The example uses a D3Q27 lattice, which is a three-dimensional lattice model that considers 
    27 discrete velocity directions. This allows for a more accurate representation of the fluid flow 
    in three dimensions.

2. NACA Airfoil Generation: The example includes a function to generate a NACA airfoil shape, which is 
    common in aerodynamics. The function allows for customization of the length, thickness, and angle 
    of the airfoil.

3. Boundary Conditions: The example includes several boundary conditions. These include a "bounce back" 
    condition on the airfoil surface and the top and bottom of the domain, a "do nothing" condition 
    at the outlet (right side of the domain), and an "equilibrium" condition at the inlet 
    (left side of the domain) to simulate a uniform flow.

4. Simulation Parameters: The example allows for the setting of various simulation parameters, 
    including the Reynolds number, inlet velocity, and characteristic length. 

5. Visualization: The example outputs data in VTK format, which can be visualized using software such 
    as Paraview. The error between the old and new velocity fields is also printed out at each time step 
    to monitor the convergence of the solution.
"""


import numpy as np
# from IPython import display
import matplotlib.pylab as plt
from src.models import BGKSim, KBCSim
from src.lattice import LatticeD3Q19, LatticeD3Q27
from src.boundary_conditions import *
import numpy as np
from src.utils import *
from jax.config import config
import os
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
import jax
import jax.numpy as jnp
import scipy

import phantomgaze as pg

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


# Function to create a NACA airfoil shape given its length, thickness, and angle of attack
def makeNacaAirfoil(length, thickness=30, angle=0):
    def nacaAirfoil(x, thickness, chordLength):
        coeffs = [0.2969, -0.1260, -0.3516, 0.2843, -0.1015]
        exponents = [0.5, 1, 2, 3, 4]
        af = [coeff * (x / chordLength) ** exp for coeff, exp in zip(coeffs, exponents)]
        return 5. * thickness / 100 * chordLength * np.sum(af)

    x = np.arange(length)
    y = np.arange(-int(length * thickness / 200), int(length * thickness / 200))
    xx, yy = np.meshgrid(x, y)
    domain = np.where(np.abs(yy) < nacaAirfoil(xx, thickness, length), 1, 0).T

    domain = scipy.ndimage.rotate(np.rot90(domain), -angle)
    domain = np.where(domain > 0.5, 1, 0)

    return domain

class Airfoil(KBCSim):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_boundary_conditions(self):
        tx, ty = np.array([self.nx, self.ny], dtype=int) - airfoil.shape

        airfoil_mask = np.pad(airfoil, ((tx // 3, tx - tx // 3), (ty // 2, ty - ty // 2)), 'constant', constant_values=False)
        airfoil_mask = np.repeat(airfoil_mask[:, :, np.newaxis], self.nz, axis=2)
        
        airfoil_indices = np.argwhere(airfoil_mask)
        wall = np.concatenate((airfoil_indices,
                               self.boundingBoxIndices['bottom'], self.boundingBoxIndices['top']))
        self.boundary = jnp.zeros((self.nx, self.ny, self.nz), dtype=jnp.float32)
        self.boundary = self.boundary.at[tuple(wall.T)].set(1.0)
        self.boundary = self.boundary[2:-2, 2:-2, 2:-2]
        self.BCs.append(BounceBack(tuple(wall.T), self.gridInfo, self.precisionPolicy))

        doNothing = self.boundingBoxIndices['right']
        self.BCs.append(DoNothing(tuple(doNothing.T), self.gridInfo, self.precisionPolicy))

        inlet = self.boundingBoxIndices['left']
        rho_inlet = np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        vel_inlet = np.zeros((inlet.shape), dtype=self.precisionPolicy.compute_dtype)

        vel_inlet[:, 0] = prescribed_vel
        self.BCs.append(EquilibriumBC(tuple(inlet.T), self.gridInfo, self.precisionPolicy, rho_inlet, vel_inlet))

    def output_data(self, **kwargs):
        # Compute q-criterion and vorticity using finite differences
        # Get velocity field
        u = kwargs['u'][..., 1:-1, :]

        # vorticity and q-criterion
        norm_mu, q = q_criterion(u)

        # Make phantomgaze volume
        dx = 0.01
        origin = (0.0, 0.0, 0.0)
        upper_bound = (self.boundary.shape[0] * dx, self.boundary.shape[1] * dx, self.boundary.shape[2] * dx)
        q_volume = pg.objects.Volume(
            q,
            spacing=(dx, dx, dx),
            origin=origin,
        )
        norm_mu_volume = pg.objects.Volume(
            norm_mu,
            spacing=(dx, dx, dx),
            origin=origin,
        )
        boundary_volume = pg.objects.Volume(
            self.boundary,
            spacing=(dx, dx, dx),
            origin=origin,
        )

        # Make colormap for norm_mu
        colormap = pg.Colormap("jet", vmin=0.0, vmax=0.05)

        # Get camera parameters
        focal_point = (self.boundary.shape[0] * dx / 2, self.boundary.shape[1] * dx / 2, self.boundary.shape[2] * dx / 2)
        radius = 3.0
        angle = kwargs['timestep'] * 0.0001
        camera_position = (focal_point[0] + radius * np.sin(angle), focal_point[1], focal_point[2] + radius * np.cos(angle))

        # Rotate camera 
        camera = pg.Camera(position=camera_position, focal_point=focal_point, view_up=(0.0, 1.0, 0.0), max_depth=30.0, height=1080, width=1920)

        # Make wireframe
        screen_buffer = pg.render.wireframe(lower_bound=origin, upper_bound=upper_bound, thickness=0.01, camera=camera)

        # Render axes
        screen_buffer = pg.render.axes(size=0.1, center=(0.0, 0.0, 1.1), camera=camera, screen_buffer=screen_buffer)

        # Render q-criterion
        screen_buffer = pg.render.contour(q_volume, threshold=0.00003, color=norm_mu_volume, colormap=colormap, camera=camera, screen_buffer=screen_buffer)

        # Render boundary
        boundary_colormap = pg.Colormap("Greys_r", vmin=0.0, vmax=3.0, opacity=np.linspace(0.0, 6.0, 256)) # This will make it grey
        screen_buffer = pg.render.volume(boundary_volume, camera=camera, colormap=boundary_colormap, screen_buffer=screen_buffer)

        # Show the rendered image
        plt.imsave('q_criterion_' + str(kwargs['timestep']).zfill(7) + '.png', np.minimum(screen_buffer.image.get(), 1.0))

        """
        print(type(kwargs['rho']))
        rho = np.array(kwargs['rho'][..., 1:-1, :])
        u = np.array(kwargs['u'][..., 1:-1, :])
        timestep = kwargs['timestep']
        u_prev = kwargs['u_prev'][..., 1:-1, :]

        u_old = np.linalg.norm(u_prev, axis=2)
        u_new = np.linalg.norm(u, axis=2)

        err = np.sum(np.abs(u_old - u_new))
        print('error= {:07.6f}'.format(err))
        # save_image(timestep, rho, u)
        fields = {"rho": rho[..., 0], "u_x": u[..., 0], "u_y": u[..., 1], "u_z": u[..., 2]}
        save_fields_vtk(timestep, fields)
        """

if __name__ == '__main__':
    airfoil_length = 101
    airfoil_thickness = 30
    airfoil_angle = 20
    airfoil = makeNacaAirfoil(length=airfoil_length, thickness=airfoil_thickness, angle=airfoil_angle).T
    precision = 'f32/f32'

    lattice = LatticeD3Q27(precision)

    nx = airfoil.shape[0]
    ny = airfoil.shape[1]

    ny = 3 * ny
    nx = 4 * nx
    nz = 101

    Re = 10000.0
    prescribed_vel = 0.1
    clength = airfoil_length

    visc = prescribed_vel * clength / Re
    omega = 1.0 / (3. * visc + 0.5)
    
    os.system('rm -rf ./*.vtk && rm -rf ./*.png')

    # Set the parameters for the simulation
    kwargs = {
        'lattice': lattice,
        'omega': omega,
        'nx': nx,
        'ny': ny,
        'nz': nz,
        'precision': precision,
        'io_rate': 10,
        'print_info_rate': 100,
    }

    sim = Airfoil(**kwargs)
    sim.run(20000)
