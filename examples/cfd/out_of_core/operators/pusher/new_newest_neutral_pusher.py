import warp as wp

from pumpkin_pulse.struct.field import Fieldfloat32
from pumpkin_pulse.struct.particles import Particles, Particle
from pumpkin_pulse.struct.material_properties import MaterialProperties
from pumpkin_pulse.operator.pusher.pusher import Pusher
from pumpkin_pulse.functional.indexing import pos_to_cell_index, pos_to_lower_cell_index
from pumpkin_pulse.functional.marching_cube import get_triangles_in_cell, _solid_fractions_to_index, _vertex_table, _triangle_table
from pumpkin_pulse.functional.ray_triangle_intersect import ray_triangle_intersect


class NeutralPusher(Pusher):
    """
    Pushes particles that have no charge in time
    """

    def __init__(
        self,
        boundary_conditions: str = "periodic",
    ):
        super().__init__(boundary_conditions)

        @wp.func
        def get_triangles(
            lower_cell_index: wp.vec3i,
            material_properties: MaterialProperties,
        ):

            # Get origin and spacing of 8 corners
            origin = (
                material_properties.id.origin
                + wp.cw_mul(
                    material_properties.id.spacing,
                    (
                        wp.vec3(wp.float32(lower_cell_index[0]), wp.float32(lower_cell_index[1]), wp.float32(lower_cell_index[2]))
                    ),
                )
                + (wp.float32(0.0) * material_properties.id.spacing)
            )

            # Get cell id on 8 corners
            id_000 = material_properties.id.data[lower_cell_index[0], lower_cell_index[1], lower_cell_index[2]]
            id_100 = material_properties.id.data[lower_cell_index[0] + 1, lower_cell_index[1], lower_cell_index[2]]
            id_010 = material_properties.id.data[lower_cell_index[0], lower_cell_index[1] + 1, lower_cell_index[2]]
            id_110 = material_properties.id.data[lower_cell_index[0] + 1, lower_cell_index[1] + 1, lower_cell_index[2]]
            id_001 = material_properties.id.data[lower_cell_index[0], lower_cell_index[1], lower_cell_index[2] + 1]
            id_101 = material_properties.id.data[lower_cell_index[0] + 1, lower_cell_index[1], lower_cell_index[2] + 1]
            id_011 = material_properties.id.data[lower_cell_index[0], lower_cell_index[1] + 1, lower_cell_index[2] + 1]
            id_111 = material_properties.id.data[lower_cell_index[0] + 1, lower_cell_index[1] + 1, lower_cell_index[2] + 1]

            # Get cell solid fraction on 8 corners
            sf_000 = material_properties.solid_fraction_mapping[warp.int32(id_000)]
            sf_100 = material_properties.solid_fraction_mapping[warp.int32(id_100)]
            sf_010 = material_properties.solid_fraction_mapping[warp.int32(id_010)]
            sf_110 = material_properties.solid_fraction_mapping[warp.int32(id_110)]
            sf_001 = material_properties.solid_fraction_mapping[warp.int32(id_001)]
            sf_101 = material_properties.solid_fraction_mapping[warp.int32(id_101)]
            sf_011 = material_properties.solid_fraction_mapping[warp.int32(id_011)]
            sf_111 = material_properties.solid_fraction_mapping[warp.int32(id_111)]

            # Get triangles in cell
            triangles, num_triangles = get_triangles_in_cell(
                sf_000,
                sf_100,
                sf_010,
                sf_110,
                sf_001,
                sf_101,
                sf_011,
                sf_111,
                origin,
                material_properties.id.spacing,
            )

            return triangles, num_triangles

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

            # Get particle data
            d = particles.data[i]

            # Get particles momentum and position
            mom = d.momentum
            pos = d.position

            # Get cell index
            center_cell_index = pos_to_cell_index(
                pos,
                particles.cell_particle_mapping.origin,
                particles.cell_particle_mapping.spacing,
                particles.cell_particle_mapping.nr_ghost_cells
                #material_properties.id.origin,
                #material_properties.id.spacing,
                #material_properties.id.nr_ghost_cells
            )
            cell_index_000 = wp.vec3i(center_cell_index[0] - 1, center_cell_index[1] - 1, center_cell_index[2] - 1)
            cell_index_100 = wp.vec3i(center_cell_index[0], center_cell_index[1] - 1, center_cell_index[2] - 1)
            cell_index_010 = wp.vec3i(center_cell_index[0] - 1, center_cell_index[1], center_cell_index[2] - 1)
            cell_index_110 = wp.vec3i(center_cell_index[0], center_cell_index[1], center_cell_index[2] - 1)
            cell_index_001 = wp.vec3i(center_cell_index[0] - 1, center_cell_index[1] - 1, center_cell_index[2])
            cell_index_101 = wp.vec3i(center_cell_index[0], center_cell_index[1] - 1, center_cell_index[2])
            cell_index_011 = wp.vec3i(center_cell_index[0] - 1, center_cell_index[1], center_cell_index[2])
            cell_index_111 = wp.vec3i(center_cell_index[0], center_cell_index[1], center_cell_index[2])

            # Get triangles in cell
            triangles_000, num_triangles_000 = get_triangles(
                cell_index_000,
                material_properties,
            )
            triangles_100, num_triangles_100 = get_triangles(
                cell_index_100,
                material_properties,
            )
            triangles_010, num_triangles_010 = get_triangles(
                cell_index_010,
                material_properties,
            )
            triangles_110, num_triangles_110 = get_triangles(
                cell_index_110,
                material_properties,
            )
            triangles_001, num_triangles_001 = get_triangles(
                cell_index_001,
                material_properties,
            )
            triangles_101, num_triangles_101 = get_triangles(
                cell_index_101,
                material_properties,
            )
            triangles_011, num_triangles_011 = get_triangles(
                cell_index_011,
                material_properties,
            )
            triangles_111, num_triangles_111 = get_triangles(
                cell_index_111,
                material_properties,
            )

            # Store remaining time to push particle
            remaining_dt = dt

            # Allocate pushed particle
            pushed_particle = Particle(
                pos,
                mom,
                wp.uint8(0), # Keep particle
            )

            # Move particle until remaining time is zero
            for _i in range(10): # Maximum number of pushes

                # Get velocity
                v = mom / particles.mass

                # Set push time
                push_dt = remaining_dt

                # Push particle
                new_pos = pos + push_dt * v

                # Check if particle hits a boundary
                hit = wp.bool(False)

                # Find intersection triangle_000
                for j_000 in range(num_triangles_000):

                    # Ray triangle intersection
                    new_pos, t = ray_triangle_intersect(
                        pos,
                        new_pos,
                        wp.vec3(triangles_000[j_000, 0], triangles_000[j_000, 1], triangles_000[j_000, 2]),
                        wp.vec3(triangles_000[j_000, 3], triangles_000[j_000, 4], triangles_000[j_000, 5]),
                        wp.vec3(triangles_000[j_000, 6], triangles_000[j_000, 7], triangles_000[j_000, 8]),
                    )

                    # Check if hit
                    if t < 1.0:
                        push_dt = push_dt * t
                        hit = wp.bool(True)

                # Find intersection triangle_100
                for j_100 in range(num_triangles_100):

                    # Ray triangle intersection
                    new_pos, t = ray_triangle_intersect(
                        pos,
                        new_pos,
                        wp.vec3(triangles_100[j_100, 0], triangles_100[j_100, 1], triangles_100[j_100, 2]),
                        wp.vec3(triangles_100[j_100, 3], triangles_100[j_100, 4], triangles_100[j_100, 5]),
                        wp.vec3(triangles_100[j_100, 6], triangles_100[j_100, 7], triangles_100[j_100, 8]),
                    )

                    # Check if hit
                    if t < 1.0:
                        push_dt = push_dt * t
                        hit = wp.bool(True)

                # Find intersection triangle_010
                for j_010 in range(num_triangles_010):

                    # Ray triangle intersection
                    new_pos, t = ray_triangle_intersect(
                        pos,
                        new_pos,
                        wp.vec3(triangles_010[j_010, 0], triangles_010[j_010, 1], triangles_010[j_010, 2]),
                        wp.vec3(triangles_010[j_010, 3], triangles_010[j_010, 4], triangles_010[j_010, 5]),
                        wp.vec3(triangles_010[j_010, 6], triangles_010[j_010, 7], triangles_010[j_010, 8]),
                    )

                    # Check if hit
                    if t < 1.0:
                        push_dt = push_dt * t
                        hit = wp.bool(True)

                # Find intersection triangle_110
                for j_110 in range(num_triangles_110):

                    # Ray triangle intersection
                    new_pos, t = ray_triangle_intersect(
                        pos,
                        new_pos,
                        wp.vec3(triangles_110[j_110, 0], triangles_110[j_110, 1], triangles_110[j_110, 2]),
                        wp.vec3(triangles_110[j_110, 3], triangles_110[j_110, 4], triangles_110[j_110, 5]),
                        wp.vec3(triangles_110[j_110, 6], triangles_110[j_110, 7], triangles_110[j_110, 8]),
                    )

                    # Check if hit
                    if t < 1.0:
                        push_dt = push_dt * t
                        hit = wp.bool(True)

                # Find intersection triangle_001
                for j_001 in range(num_triangles_001):

                    # Ray triangle intersection
                    new_pos, t = ray_triangle_intersect(
                        pos,
                        new_pos,
                        wp.vec3(triangles_001[j_001, 0], triangles_001[j_001, 1], triangles_001[j_001, 2]),
                        wp.vec3(triangles_001[j_001, 3], triangles_001[j_001, 4], triangles_001[j_001, 5]),
                        wp.vec3(triangles_001[j_001, 6], triangles_001[j_001, 7], triangles_001[j_001, 8]),
                    )

                    # Check if hit
                    if t < 1.0:
                        push_dt = push_dt * t
                        hit = wp.bool(True)

                # Find intersection triangle_101
                for j_101 in range(num_triangles_101):

                    # Ray triangle intersection
                    new_pos, t = ray_triangle_intersect(
                        pos,
                        new_pos,
                        wp.vec3(triangles_101[j_101, 0], triangles_101[j_101, 1], triangles_101[j_101, 2]),
                        wp.vec3(triangles_101[j_101, 3], triangles_101[j_101, 4], triangles_101[j_101, 5]),
                        wp.vec3(triangles_101[j_101, 6], triangles_101[j_101, 7], triangles_101[j_101, 8]),
                    )

                    # Check if hit
                    if t < 1.0:
                        push_dt = push_dt * t
                        hit = wp.bool(True)

                # Find intersection triangle_011
                for j_011 in range(num_triangles_011):

                    # Ray triangle intersection
                    new_pos, t = ray_triangle_intersect(
                        pos,
                        new_pos,
                        wp.vec3(triangles_011[j_011, 0], triangles_011[j_011, 1], triangles_011[j_011, 2]),
                        wp.vec3(triangles_011[j_011, 3], triangles_011[j_011, 4], triangles_011[j_011, 5]),
                        wp.vec3(triangles_011[j_011, 6], triangles_011[j_011, 7], triangles_011[j_011, 8]),
                    )

                    # Check if hit
                    if t < 1.0:
                        push_dt = push_dt * t
                        hit = wp.bool(True)

                # Find intersection triangle_111
                for j_111 in range(num_triangles_111):

                    # Ray triangle intersection
                    new_pos, t = ray_triangle_intersect(
                        pos,
                        new_pos,
                        wp.vec3(triangles_111[j_111, 0], triangles_111[j_111, 1], triangles_111[j_111, 2]),
                        wp.vec3(triangles_111[j_111, 3], triangles_111[j_111, 4], triangles_111[j_111, 5]),
                        wp.vec3(triangles_111[j_111, 6], triangles_111[j_111, 7], triangles_111[j_111, 8]),
                    )

                    # Check if hit
                    if t < 1.0:
                        push_dt = push_dt * t
                        hit = wp.bool(True)

                # Check if hit
                if hit:

                    # Reverse momentum
                    mom = -mom
                    new_pos = pos

                # Update remaining time
                remaining_dt -= push_dt

                # Update particle position
                pos = new_pos

                # Check if remaining time is zero
                if remaining_dt <= 0.0:
                    pushed_particle.position = pos
                    pushed_particle.momentum = mom
                    pushed_particle.kill = wp.uint8(0) # Keep particle
                    break

                # Checki if maximum number of pushes is reached
                if _i == 9:
                    print("Particle pushed too many times")
                    pushed_particle.position = pos
                    pushed_particle.momentum = mom
                    pushed_particle.kill = wp.uint8(1)

            # Set new position
            particles.data[i] = pushed_particle

            # Get index of new cell
            ijk = pos_to_cell_index(
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

        # Update cell particle mapping
        wp.utils.array_scan(
            particles.cell_particle_mapping_buffer.data,
            particles.cell_particle_mapping.data,
            inclusive=False,
        )

        # Sort particles
        particles = Pusher.sort_particles(particles)

        return particles
