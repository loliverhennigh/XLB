import warp as wp

from pumpkin_pulse.struct.particles import Particles
from pumpkin_pulse.operator.operator import Operator

class ParticleSorter(Operator):

    @wp.kernel
    def _sort_particles(
        particles: Particles,
    ):
        # Get particle index
        i = wp.tid()

        # Load particle data
        pos = particles.position[i]
        mom = particles.momentum[i]
        k = particles.kill[i]

        # Skip if particle is dead
        if k == wp.uint8(1):
            wp.atomic_add(particles.nr_particles, 0, -1)
            return

        # Determine cell index
        ijk_f = wp.cw_div((pos - particles.cell_particle_mapping.origin), particles.cell_particle_mapping.spacing)
        ijk = wp.vec3i(
            wp.int32(wp.floor(ijk_f[0])) + particles.cell_particle_mapping.nr_ghost_cells,
            wp.int32(wp.floor(ijk_f[1])) + particles.cell_particle_mapping.nr_ghost_cells,
            wp.int32(wp.floor(ijk_f[2])) + particles.cell_particle_mapping.nr_ghost_cells,
        )

        # Get cell particle mapping index
        particle_index = particles.cell_particle_mapping.data[ijk[0], ijk[1], ijk[2]]

        # Get offset with buffer
        offset = wp.atomic_add(particles.cell_particle_mapping_buffer.data, ijk[0], ijk[1], ijk[2], 1)

        # Add particle to cell
        particles.position_buffer[offset + particle_index] = pos
        particles.momentum_buffer[offset + particle_index] = mom
        particles.kill_buffer[offset + particle_index] = k

    def __call__(
        self,
        particles: Particles,
    ):

        # Zero cell particle mapping buffer
        particles.cell_particle_mapping_buffer.data.zero_()

        # Compute number of particles to add each cell
        wp.launch(
            self._sort_particles,
            inputs=[
                particles,
            ],
            dim=(particles.nr_particles.numpy()[0],),
        )

        # Rotate buffers
        particles.position, particles.position_buffer = particles.position_buffer, particles.position
        particles.momentum, particles.momentum_buffer = particles.momentum_buffer, particles.momentum
        particles.kill, particles.kill_buffer = particles.kill_buffer, particles.kill

        return particles
