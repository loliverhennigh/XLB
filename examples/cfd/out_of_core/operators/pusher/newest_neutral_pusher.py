import warp as wp

from pumpkin_pulse.struct.field import Fieldfloat32
from pumpkin_pulse.struct.particles import Particles, Particle
from pumpkin_pulse.struct.material_properties import MaterialProperties
from pumpkin_pulse.operator.pusher.pusher import Pusher


class NeutralPusher(Pusher):
    """
    Pushes particles that have no charge in time
    """

    def __init__(
        self,
        boundary_conditions: str = "periodic",
    ):
        super().__init__(boundary_conditions)

        # Make push kernel
        @wp.kernel
        def push(
            particles: Particles,
            material_properties: MaterialProperties,
            temperature: Fieldfloat32,
            dt: wp.float32,
        ):

            # Get particle index
            i = wp.tid()

            # Get particle data
            d = particles.data[i]

            # Get particles momentum and position
            mom = d.momentum
            pos = d.position

            # Store remaining time to push particle
            remaining_dt = dt

            # Allocate pushed particle
            pushed_particle = Particle(
                pos,
                mom,
                wp.uint8(0), # Keep particle
            )

            # Get epsilon distance for shell around solid
            epsilon = particles.cell_particle_mapping.spacing[0] * 1e-2

            # Get stencils
            solid_fraction_stencil, pos_stencil = Pusher.get_solid_fraction_stencil(
                pos,
                material_properties,
            )

            # Hit on previous time step
            hit = wp.bool(False)

            # Move particle until remaining time is zero
            for _ in range(100): # Maximum number of pushes

                # Get sdf for current position
                sdf = Pusher.solid_fraction_stencil_to_sdf(
                    pos,
                    solid_fraction_stencil,
                    pos_stencil,
                    material_properties.id.spacing,
                )

                # Get velocity
                v = mom / particles.mass

                # Compute maximum push time
                if not hit:
                    push_dt = wp.min(
                        sdf / wp.length(v), # Time to solid
                        remaining_dt, # Remaining time
                    )
                else:
                    push_dt = epsilon / wp.length(v) # Small push to get out of solid

                # Push particle
                new_pos = pos + push_dt * v

                # Update remaining time
                remaining_dt -= push_dt

                # Get sdf for new position
                new_sdf = Pusher.solid_fraction_stencil_to_sdf(
                    new_pos,
                    solid_fraction_stencil,
                    pos_stencil,
                    material_properties.id.spacing,
                )

                # Check if particle hit solid
                if (new_sdf <= epsilon) and (not hit):

                    # Get cell index
                    cell_index = Pusher.pos_to_cell_index(
                        new_pos,
                        particles.cell_particle_mapping.origin,
                        particles.cell_particle_mapping.spacing,
                        particles.cell_particle_mapping.nr_ghost_cells
                    )

                    # Get solid id
                    solid_id = material_properties.id.data[
                        cell_index[0],
                        cell_index[1],
                        cell_index[2],
                    ]

                    # Get solid type
                    solid_type = material_properties.solid_type_mapping[warp.int32(solid_id)]

                    # Get temperature
                    temp = temperature.data[cell_index[0], cell_index[1], cell_index[2]]

                    # Store enegy change
                    energy_change = wp.float32(0.0)


                    # 0: Reflective
                    if solid_type == 0:

                        # Get normal
                        normal = Pusher.solid_fraction_stencil_to_sdf_gradient(
                            new_pos,
                            solid_fraction_stencil,
                            pos_stencil,
                            material_properties.id.spacing,
                        )

                        # Reflect momentum
                        mom = mom - 2.0 * wp.dot(mom, normal) * normal

                    # 1: Absorbing
                    elif solid_type == 1:

                        # Kill particle
                        pushed_particle.position = new_pos
                        pushed_particle.momentum = wp.vec3(0.0, 0.0, 0.0)
                        pushed_particle.kill = wp.uint8(1)
                        break

                    # 2: Stop
                    elif solid_type == 2:

                        # Stop particle but keep it
                        pushed_particle.position = new_pos
                        pushed_particle.momentum = wp.vec3(0.0, 0.0, 0.0)
                        pushed_particle.kill = wp.uint8(0)
                        break

                    # 3: Thermalize
                    elif solid_type == 3:

                        # Get normal
                        normal = Pusher.solid_fraction_stencil_to_sdf_gradient(
                            new_pos,
                            solid_fraction_stencil,
                            pos_stencil,
                            material_properties.id.spacing,
                        )

                        # Get temperature

                        # Reflect momentum
                        mom = mom - 2.0 * wp.dot(mom, normal) * normal

                        # Add thermal velocity
                        mom = mom + wp.sqrt(2.0 * wp.k_b * temperature.data[0] / particles.mass) * wp.vec3(
                            wp.randn(),
                            wp.randn(),
                            wp.randn(),
                        )

                    # Get normal
                    normal = Pusher.solid_fraction_stencil_to_sdf_gradient(
                        new_pos,
                        solid_fraction_stencil,
                        pos_stencil,
                        material_properties.id.spacing,
                    )

                    # Reflect momentum
                    mom = mom - 2.0 * wp.dot(mom, normal) * normal

                    # Set hit to true
                    hit = wp.bool(True)

                # Check if particle is outside solid
                if new_sdf > epsilon:
                    hit = wp.bool(False)

                # Update position
                pos = new_pos

                # Check if particle is inside solid
                if new_sdf < 0.0:
                    print("Particle inside solid")

                # Check if maximum number of pushes is reached
                if _ == 99:
                    print("Maximum number of pushes reached")

                # Check if particle needs to be pushed
                if remaining_dt <= 0.0:
                    pushed_particle.position = pos
                    pushed_particle.momentum = mom
                    pushed_particle.kill = wp.uint8(0)
                    break

            # Set new position
            particles.data[i] = pushed_particle

            # Get index of new cell
            ijk = Pusher.pos_to_cell_index(
                pushed_particle.position,
                particles.cell_particle_mapping.origin,
                particles.cell_particle_mapping.spacing,
                particles.cell_particle_mapping.nr_ghost_cells
            )

            # Add particle count to particles per cell
            wp.atomic_add(particles.cell_particle_mapping_buffer.data, ijk[0], ijk[1], ijk[2], 1)

        # Store push kernel
        self.push = push

    def __call__(
        self,
        particles: Particles,
        material_properties: MaterialProperties,
        temperature: Fieldfloat32,
        dt: float,
    ):

        # Zero cell particle mapping buffer
        particles.cell_particle_mapping_buffer.data.zero_()

        # Compute number of particles to add each cel
        wp.launch(
            self.push,
            inputs=[
                particles,
                material_properties,
                temperature,
                dt,
            ],
            dim=(particles.nr_particles.numpy()[0],),
        )

        # Update cell particle mapping
        wp.utils.array_scan(
            particles.cell_particle_mapping_buffer.data,
            particles.cell_particle_mapping.data,
            inclusive=False,
        )

        # Sort particles
        particles = Pusher.sort_particles(particles)

        return particles
