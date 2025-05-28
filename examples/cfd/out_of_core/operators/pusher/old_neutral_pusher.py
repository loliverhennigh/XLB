import warp as wp

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
            dt: wp.float32,
        ):

            # Get particle index
            i = wp.tid()

            # Get particle data
            d = particles.data[i]

            # Get particles current momentum and position
            current_mom = d.momentum
            current_pos = d.position

            # Store remaining time to push particle
            remaining_dt = dt

            # Allocate pushed particle
            pushed_particle = Particle(
                current_pos,
                current_mom,
                wp.uint8(0), # Keep particle
            )

            # Move particle through cell walls and check for solid interactions
            for _ in range(100): # 100 as maximum number of steps to take (should never be reached)

                # Check if particle needs to be pushed
                if remaining_dt <= 0.0:
                    pushed_particle.position = current_pos
                    pushed_particle.momentum = current_mom
                    pushed_particle.kill = wp.uint8(0)
                    break

                # Compute the minimum timestep particle can travel in a cell
                push_dt = Pusher.min_timestep(
                    current_pos,
                    current_mom / particles.mass,
                    remaining_dt,
                    particles.cell_particle_mapping.origin,
                    particles.cell_particle_mapping.spacing,
                )

                # Push particle
                new_pos = current_pos + push_dt * current_mom / particles.mass

                # Get lower cell index of new position
                lower_cell_index = Pusher.pos_to_lower_cell_index(
                    new_pos,
                    material_properties.id.origin,
                    material_properties.id.spacing,
                    material_properties.id.nr_ghost_cells,
                )

                # Get id for all corners of cell
                id_corners = Pusher.get_id_corners(
                    lower_cell_index,
                    material_properties,
                )

                # Get solid fraction for new position
                sf_corners = Pusher.get_solid_fraction_corners(
                    id_corners,
                    material_properties,
                )

                # Get relative position
                new_relative_pos = Pusher.pos_to_relative_pos(
                    new_pos,
                    particles.cell_particle_mapping.origin,
                    particles.cell_particle_mapping.spacing,
                )

                # Get solid fraction at new position
                new_sf = Pusher.interpolate_solid_fraction_to_pos(
                    sf_corners,
                    new_relative_pos
                )

                # Particle is in vacuum
                if new_sf <= 0.501:
                    current_pos = new_pos
                    remaining_dt = remaining_dt - push_dt

                # Particle is in solid
                else:
                    # Get relative position of current position
                    relative_current_pos = Pusher.pos_to_relative_pos(
                        current_pos,
                        particles.cell_particle_mapping.origin,
                        particles.cell_particle_mapping.spacing,
                    )

                    # Get the solid fraction at the current position
                    current_sf = Pusher.interpolate_solid_fraction_to_pos(
                        sf_corners,
                        relative_current_pos
                    )

                    # Find time of intersection
                    push_dt = push_dt * (0.501 - current_sf) / (new_sf - current_sf)

                    # Move particle to new position
                    current_pos = current_pos + push_dt * current_mom / particles.mass

                    # Get remaining time to push particle
                    remaining_dt = remaining_dt - push_dt

                    # Get relative position again
                    relative_current_pos = Pusher.pos_to_relative_pos(
                        current_pos,
                        particles.cell_particle_mapping.origin,
                        particles.cell_particle_mapping.spacing,
                    )

                    # Get wall kind for current position
                    k = Pusher.interpolate_solid_type_to_pos(
                        id_corners,
                        material_properties,
                        relative_current_pos
                    )

                    # Apply different wall kinds
                    # Reflective
                    if k == wp.uint8(0):
                        # Get normal at wall
                        normal_sf = Pusher.interpolate_normal_solid_fraction_to_pos(
                            sf_corners,
                        )

                        # Reflect momentum
                        #current_mom = current_mom - 2.0 * wp.dot(current_mom, normal_sf) * normal_sf
                        current_mom = -current_mom

                    # Absorbing
                    elif k == wp.uint8(1):
                        pushed_particle.position = current_pos
                        pushed_particle.momentum = wp.vec3f(0.0, 0.0, 0.0)
                        pushed_particle.kill = wp.uint8(1) # Kill particle
                        break

                    # Stopping
                    elif k == wp.uint8(2):
                        pushed_particle.position = current_pos
                        pushed_particle.momentum = wp.vec3f(0.0, 0.0, 0.0)
                        pushed_particle.kill = wp.uint8(0)
                        break

                # Check if last step
                if _ == 99:
                    print("Warning: Maximum number of steps reached")
                    pushed_particle.position = current_pos
                    pushed_particle.momentum = current_mom
                    pushed_particle.kill = wp.uint8(0)

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
