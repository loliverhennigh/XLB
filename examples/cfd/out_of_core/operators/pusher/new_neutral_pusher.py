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
            id_stencil, solid_fraction_stencil, solid_type_stencil, pos_stencil = Pusher.get_stencils(
                pos,
                material_properties,
            )

            # Move particle until remaining time is zero
            for _ in range(1000): # Maximum number of pushes

                # Get sdf for current position
                sdf = Pusher.solid_fraction_sdf(
                    pos,
                    solid_fraction_stencil,
                    pos_stencil,
                    material_properties.id.spacing,
                )

                # Get velocity
                v = mom / particles.mass

                # 2 Cases:
                # 1. Particle starts on outside of solid (sdf > 0)
                # 3. Particle starts on inside of solid (sdf <= 0)
                if sdf > 0.0: # Case 1

                    # Compute maximum push time
                    push_dt = wp.min(
                        wp.max(
                            (sdf - epsilon) / wp.length(v), # Time to outer solid shell
                            epsilon / wp.length(v), # If inside solid shell then push epsilon distance
                        ),
                        remaining_dt, # Remaining time
                    )

                    # Push particle
                    new_pos = pos + push_dt * v

                    # Get sdf for new position
                    new_sdf = Pusher.solid_fraction_sdf(
                        new_pos,
                        solid_fraction_stencil,
                        pos_stencil,
                        material_properties.id.spacing,
                    )

                    # 2 Cases:
                    # 1. Particle remains on outside or enters outer solid shell
                    # 2. Particle enters solid
                    if new_sdf > 0.0: # Case 1

                        # Update remaining time
                        remaining_dt -= push_dt

                        # Update position
                        pos = new_pos

                    else: # Case 2

                        ## Iterate to get intersection on solid shell
                        #for _i in range(10): # Maximum number of iterations

                        # Linear interpolate to get intersection on solid shell
                        push_dt = push_dt * sdf / (sdf - new_sdf)
                        new_pos = pos + push_dt * v

                        ## Get sdf for new position
                        #new_sdf = Pusher.solid_fraction_sdf(
                        #    new_pos,
                        #    solid_fraction_stencil,
                        #    pos_stencil,
                        #    material_properties.id.spacing,
                        #)

                        #if wp.abs(new_sdf) > 1e-6:
                        #    print("Error in intersection")
                        #    print(sdf)
                        #    print(new_sdf)
                        #    print(_)

                        # Get normal
                        normal = Pusher.solid_fraction_normal(
                            new_pos,
                            solid_fraction_stencil,
                            pos_stencil,
                            material_properties.id.spacing,
                        )

                        # Reflect momentum
                        mom = mom - 2.0 * wp.dot(mom, normal) * normal

                        # Update remaining time
                        remaining_dt -= push_dt

                        # Update position
                        pos = new_pos

                else: # Case 2

                    # Compute maximum push time to get to inner solid shell
                    push_dt = wp.min(epsilon / wp.length(v), remaining_dt)

                    # Push particle
                    new_pos = pos + push_dt * v

                    # Get sdf for new position
                    new_sdf = Pusher.solid_fraction_sdf(
                        new_pos,
                        solid_fraction_stencil,
                        pos_stencil,
                        material_properties.id.spacing,
                    )

                    # 2 Cases:
                    # 1. Particle goes further into solid (This should not happen)
                    # 2. Particle goes away from solid
                    if new_sdf < sdf: # Case 1

                        # Flip momentum to force particle to move away from solid
                        mom = -mom

                        # Print warning
                        print("Particle went further into solid")

                    else: # Case 2

                        # Update remaining time
                        remaining_dt -= push_dt

                        # Update position
                        pos = new_pos

                # Check if maximum number of pushes is reached
                if _ == 999:
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
