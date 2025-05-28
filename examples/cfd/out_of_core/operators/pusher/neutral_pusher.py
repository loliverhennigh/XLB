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
            vertex_indices_table: wp.array2d(dtype=wp.int32),
            vertex_table: wp.array2d(dtype=wp.float32),
        ):

            # Get particle index
            i = wp.tid()

            # Get particles momentum and position
            mom = particles.momentum[i]
            pos = particles.position[i]

            # Get cell index
            cell_index = pos_to_cell_index(
                pos,
                material_properties.mc_id.origin,
                material_properties.mc_id.spacing,
                material_properties.mc_id.nr_ghost_cells
            )

            # Get stencil of marching cubes id
            mc_id_stencil = NeutralPusher._mc_id_stencil_type()
            for _i in range(27):

                # Get index of cell
                stencil_index_i = cell_index[0] + (_i % 3) - 1
                stencil_index_j = cell_index[1] + ((_i // 3) % 3) - 1
                stencil_index_k = cell_index[2] + (_i // 9) - 1

                # Store marching cubes id
                mc_id_stencil[_i] = material_properties.mc_id.data[stencil_index_i, stencil_index_j, stencil_index_k]

            # Store remaining time to push particle
            remaining_dt = dt

            # Check if particle hits a boundary
            hit = wp.bool(False)
            nr_hits = wp.int32(0)

            # Move particle until remaining time is zero
            for _ in range(10): # Maximum number of pushes

                # Get velocity
                v = mom / particles.mass

                # Set push time
                push_dt = remaining_dt

                # Push particle
                new_pos = pos + push_dt * v

                # Reset hit
                hit = wp.bool(False)

                # Normalized distance to nearest triangle
                running_d = wp.float32(1.0)
                running_t = wp.float32(1.0)

                # Find intersection any triangles
                # Loop over all neighbor cells (3x3x3)

                for _i in range(27):

                    # Check if any triangles in cell
                    if vertex_indices_table[wp.int32(mc_id_stencil[_i]), 0] != -1:

                        # Get index of cell
                        stencil_index_i = cell_index[0] + (_i % 3) - 1
                        stencil_index_j = cell_index[1] + ((_i // 3) % 3) - 1
                        stencil_index_k = cell_index[2] + (_i // 9) - 1

                        # Get origin of cell
                        cell_origin = (
                            material_properties.mc_id.origin
                            + wp.cw_mul(
                                material_properties.mc_id.spacing,
                                (
                                    wp.vec3(wp.float32(stencil_index_i), wp.float32(stencil_index_j), wp.float32(stencil_index_k))
                                ),
                            )
                        )

                        # For each cell check collisions with triangles (max 5 triangles)
                        for _ti in range(5):

                            # Get vertex index
                            vertex_index_0 = vertex_indices_table[wp.int32(mc_id_stencil[_i]), _ti * 3 + 0]
                            vertex_index_1 = vertex_indices_table[wp.int32(mc_id_stencil[_i]), _ti * 3 + 1]
                            vertex_index_2 = vertex_indices_table[wp.int32(mc_id_stencil[_i]), _ti * 3 + 2]

                            # Break if no triangle
                            if vertex_index_0 == -1:
                                break
                            else:

                                # Get triangle
                                vertex_0 = wp.vec3(
                                    vertex_table[vertex_index_0, 0] * material_properties.mc_id.spacing[0] + cell_origin[0],
                                    vertex_table[vertex_index_0, 1] * material_properties.mc_id.spacing[1] + cell_origin[1],
                                    vertex_table[vertex_index_0, 2] * material_properties.mc_id.spacing[2] + cell_origin[2],
                                )
                                vertex_1 = wp.vec3(
                                    vertex_table[vertex_index_1, 0] * material_properties.mc_id.spacing[0] + cell_origin[0],
                                    vertex_table[vertex_index_1, 1] * material_properties.mc_id.spacing[1] + cell_origin[1],
                                    vertex_table[vertex_index_1, 2] * material_properties.mc_id.spacing[2] + cell_origin[2],
                                )
                                vertex_2 = wp.vec3(
                                    vertex_table[vertex_index_2, 0] * material_properties.mc_id.spacing[0] + cell_origin[0],
                                    vertex_table[vertex_index_2, 1] * material_properties.mc_id.spacing[1] + cell_origin[1],
                                    vertex_table[vertex_index_2, 2] * material_properties.mc_id.spacing[2] + cell_origin[2],
                                )

                                # Ray triangle intersection
                                intra_pos, t, d = ray_triangle_intersect(
                                    pos,
                                    new_pos,
                                    vertex_0,
                                    vertex_1,
                                    vertex_2,
                                    material_properties.mc_id.spacing[0],
                                )

                                # Check if hit
                                if d < running_d:

                                    # Update running distance
                                    running_d = d

                                    # Update running time
                                    running_t = t

                                    # Update hit
                                    hit = wp.bool(True)

                                    # Update normal
                                    normal = - wp.cross(vertex_1 - vertex_0, vertex_2 - vertex_0)
                                    normal = normal / wp.length(normal)

                # Check if hit
                if hit:

                    # Reflect momentum
                    if wp.dot(mom, normal) < 0.0:
                        mom = mom - 2.0 * wp.dot(mom, normal) * normal

                    # Slightly reduce push time
                    push_dt = 0.99 * running_t * push_dt
                    new_pos = pos + push_dt * v
                    nr_hits += 1

                # Update remaining time
                remaining_dt -= push_dt

                # Update particle position
                pos = new_pos

                # Check if remaining time is zero
                if remaining_dt <= 0.0:
                    break

                # Check if maximum number of pushes is reached
                if _ == 9:
                    #print("Particle pushed too many times")
                    break

            #if nr_hits > 2:
            #    print("Particle hit multiple times")
            #    print(nr_hits)

            # Get index of new cell
            ijk = pos_to_cell_index(
                pos,
                particles.cell_particle_mapping.origin,
                particles.cell_particle_mapping.spacing,
                particles.cell_particle_mapping.nr_ghost_cells
            )

            # Set new position
            particles.position[i] = pos
            particles.momentum[i] = mom

            # Check if particle is outside domain
            if ijk[0] < 1 or ijk[0] >= particles.cell_particle_mapping.shape[0]-1:
                particles.kill[i] = wp.uint8(1)
            if ijk[1] < 1 or ijk[1] >= particles.cell_particle_mapping.shape[1]-1:
                particles.kill[i] = wp.uint8(1)
            if ijk[2] < 1 or ijk[2] >= particles.cell_particle_mapping.shape[2]-1:
                particles.kill[i] = wp.uint8(1)

            # Add particle count to particles per cell
            if particles.kill[i] == 0:
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
                VERTEX_INDICES_TABLE,
                VERTEX_TABLE,
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
