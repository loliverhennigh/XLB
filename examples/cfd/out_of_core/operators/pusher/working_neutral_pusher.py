import warp as wp

from pumpkin_pulse.struct.field import Fieldfloat32
from pumpkin_pulse.struct.particles import Particles
from pumpkin_pulse.struct.material_properties import MaterialProperties
from pumpkin_pulse.operator.pusher.pusher import Pusher
from pumpkin_pulse.functional.indexing import pos_to_cell_index, pos_to_lower_cell_index
from pumpkin_pulse.functional.marching_cube import VERTEX_TABLE, VERTEX_INDICES_TABLE
from pumpkin_pulse.functional.ray_triangle_intersect import ray_triangle_intersect


class NeutralPusher(Pusher):
    """
    Pushes particles that have no charge in time
    """

    _mc_id_stencil_type = wp.vec(27, wp.uint8)
    _mc_id_origin_stencil_type = wp.mat((27, 3), wp.float32)

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
            #temperature: Fieldfloat32,
            dt: wp.float32,
        ):

            # Get particle index
            i = wp.tid()

            # Get particles momentum and position
            mom = particles.momentum[i]
            pos = particles.position[i]

            ## Get cell index
            #cell_index = pos_to_cell_index(
            #    pos,
            #    material_properties.mc_id.origin,
            #    material_properties.mc_id.spacing,
            #    material_properties.mc_id.nr_ghost_cells
            #)

            ## Get stencil of marching cubes id
            #mc_id_stencil = NeutralPusher._mc_id_stencil_type()
            #mc_id_origin_stencil = NeutralPusher._mc_id_origin_stencil_type()
            #for _i in range(27):

            #    # Get index of cell
            #    stencil_index_i = cell_index[0] + (_i % 3) - 1
            #    stencil_index_j = cell_index[1] + ((_i // 3) % 3) - 1
            #    stencil_index_k = cell_index[2] + (_i // 9) - 1

            #    # Get origin of cell
            #    stencil_origin = (
            #        material_properties.mc_id.origin
            #        + wp.cw_mul(
            #            material_properties.mc_id.spacing,
            #            (
            #                wp.vec3(wp.float32(stencil_index_i), wp.float32(stencil_index_j), wp.float32(stencil_index_k))
            #            ),
            #        )
            #    )

            #    # Store marching cubes id
            #    mc_id_stencil[_i] = material_properties.mc_id.data[stencil_index_i, stencil_index_j, stencil_index_k]

            #    # Store origin of cell
            #    mc_id_origin_stencil[_i, 0] = stencil_origin[0]
            #    mc_id_origin_stencil[_i, 1] = stencil_origin[1]
            #    mc_id_origin_stencil[_i, 2] = stencil_origin[2]

            # Store remaining time to push particle
            remaining_dt = dt

            # Move particle until remaining time is zero
            for _ in range(10): # Maximum number of pushes

                # Get velocity
                v = mom / particles.mass

                # Set push time
                push_dt = remaining_dt

                # Push particle
                new_pos = pos + push_dt * v

                # Check if particle hits a boundary
                #hit = wp.bool(False)
                #normal = wp.vec3(0.0, 0.0, 0.0)

                ## Find intersection any triangles
                ## Loop over all neighbor cells (3x3x3)
                #for _i in range(27):

                #    # For each cell check collisions with triangles (max 5 triangles)
                #    for _ti in range(5):

                #        # Get vertex index
                #        vertex_index_0 = VERTEX_INDICES_TABLE[wp.int32(mc_id_stencil[_i]), _ti * 3 + 0]
                #        vertex_index_1 = VERTEX_INDICES_TABLE[wp.int32(mc_id_stencil[_i]), _ti * 3 + 1]
                #        vertex_index_2 = VERTEX_INDICES_TABLE[wp.int32(mc_id_stencil[_i]), _ti * 3 + 2]

                #        # Break if no triangle
                #        if vertex_index_0 == -1:
                #            break
                #        else:

                #            # Get triangle
                #            vertex_0 = wp.vec3(
                #                VERTEX_TABLE[vertex_index_0, 0] * material_properties.mc_id.spacing[0] + mc_id_origin_stencil[_i, 0],
                #                VERTEX_TABLE[vertex_index_0, 1] * material_properties.mc_id.spacing[1] + mc_id_origin_stencil[_i, 1],
                #                VERTEX_TABLE[vertex_index_0, 2] * material_properties.mc_id.spacing[2] + mc_id_origin_stencil[_i, 2],
                #            )
                #            vertex_1 = wp.vec3(
                #                VERTEX_TABLE[vertex_index_1, 0] * material_properties.mc_id.spacing[0] + mc_id_origin_stencil[_i, 0],
                #                VERTEX_TABLE[vertex_index_1, 1] * material_properties.mc_id.spacing[1] + mc_id_origin_stencil[_i, 1],
                #                VERTEX_TABLE[vertex_index_1, 2] * material_properties.mc_id.spacing[2] + mc_id_origin_stencil[_i, 2],
                #            )
                #            vertex_2 = wp.vec3(
                #                VERTEX_TABLE[vertex_index_2, 0] * material_properties.mc_id.spacing[0] + mc_id_origin_stencil[_i, 0],
                #                VERTEX_TABLE[vertex_index_2, 1] * material_properties.mc_id.spacing[1] + mc_id_origin_stencil[_i, 1],
                #                VERTEX_TABLE[vertex_index_2, 2] * material_properties.mc_id.spacing[2] + mc_id_origin_stencil[_i, 2],
                #            )

                #            # Ray triangle intersection
                #            intra_pos, t = ray_triangle_intersect(
                #                pos,
                #                new_pos,
                #                vertex_0,
                #                vertex_1,
                #                vertex_2,
                #            )

                #            # Check if hit
                #            if t < 1.0:
                #                push_dt = push_dt * (0.99 * t) # Slighly reduce push time
                #                new_pos = pos + push_dt * v
                #                hit = wp.bool(True)
                #                normal = wp.normalize(wp.cross(vertex_1 - vertex_0, vertex_2 - vertex_0))

                ## Check if hit
                #if hit:

                #    # Reflect momentum
                #    mom = mom - 2.0 * wp.dot(mom, normal) * normal
                #    #new_pos = pos

                # Update remaining time
                remaining_dt -= push_dt

                # Update particle position
                pos = new_pos

                # Check if remaining time is zero
                if remaining_dt <= 0.0:
                    break

                ## Check if maximum number of pushes is reached
                #if _ == 9:
                #    print("Particle pushed too many times")
                #    pushed_particle.position = pos
                #    pushed_particle.momentum = mom
                #    pushed_particle.kill = wp.uint8(1)

            # Set new position
            particles.position[i] = pos
            particles.momentum[i] = mom
            particles.kill[i] = wp.uint8(0) # Keep particle

            # Get index of new cell
            ijk = pos_to_cell_index(
                pos,
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
        #temperature: Fieldfloat32,
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
                #temperature,
                dt,
            ],
            dim=(particles.nr_particles.numpy()[0],),
        )

        ## Update cell particle mapping
        #wp.utils.array_scan(
        #    particles.cell_particle_mapping_buffer.data,
        #    particles.cell_particle_mapping.data,
        #    inclusive=False,
        #)

        ## Sort particles
        #particles = Pusher.sort_particles(particles)

        return particles
