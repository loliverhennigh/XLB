# Base pusher class for pushing particles in time

import warp as wp

from pumpkin_pulse.struct.particles import Particles
from pumpkin_pulse.struct.material_properties import MaterialProperties
from pumpkin_pulse.operator.operator import Operator
from pumpkin_pulse.operator.sort.sort import ParticleSorter

@wp.func
def smooth_union(a: wp.float32, b: wp.float32, k: wp.float32):
    h = wp.max(k - wp.abs(a - b), 0.0)
    return wp.min(a, b) - h * h * 0.25 / k

class Pusher(Operator):
    sort_particles = ParticleSorter()

    # Define helpfull data types
    solid_fraction_stencil_type = wp.vec(125, dtype=wp.float32) # solid fraction for the 5x5x5 corners of a cell
    pos_stencil_type = wp.mat((125, 3), dtype=wp.float32) # position for the 5x5x5 corners of a cell

    def __init__(
        self,
        boundary_conditions: "periodic",
    ):

        # Set boundary condition function
        if boundary_conditions == "periodic":
            self.apply_boundary_conditions = Pusher.apply_periodic_boundary_conditions
        else:
            raise ValueError(
                f"Boundary conditions {boundary_conditions} not supported by NeutralPusher"
            )

    @wp.func
    def apply_periodic_boundary_conditions(
        pos: wp.vec3,
        origin: wp.vec3,
        spacing: wp.vec3,
        shape: wp.vec3i,
        nr_ghost_cells: wp.int32,
    ):

        # Check if particle is outside of domain
        # X-direction
        if pos[0] < origin[0]:
            pos[0] += wp.float32(shape[0] - 2 * nr_ghost_cells) * spacing[0]
        elif pos[0] >= origin[0] + wp.float32(shape[0] - 2 * nr_ghost_cells) * spacing[0]:
            pos[0] -= wp.float32(shape[0] - 2 * nr_ghost_cells) * spacing[0]

        # Y-direction
        if pos[1] < origin[1]:
            pos[1] += wp.float32(shape[1] - 2 * nr_ghost_cells) * spacing[1]
        elif pos[1] >= origin[1] + wp.float32(shape[1] - 2 * nr_ghost_cells) * spacing[1]:
            pos[1] -= wp.float32(shape[1] - 2 * nr_ghost_cells) * spacing[1]

        # Z-direction
        if pos[2] < origin[2]:
            pos[2] += wp.float32(shape[2] - 2 * nr_ghost_cells) * spacing[2]
        elif pos[2] >= origin[2] + wp.float32(shape[2] - 2 * nr_ghost_cells) * spacing[2]:
            pos[2] -= wp.float32(shape[2] - 2 * nr_ghost_cells) * spacing[2]

        return pos

    @wp.func
    def pos_to_cell_index(
        pos: wp.vec3,
        origin: wp.vec3,
        spacing: wp.vec3,
        nr_ghost_cells: wp.int32,
    ):
        float_ijk = wp.cw_div(pos - origin, spacing)
        cell_index = wp.vec3i(
            wp.int32(float_ijk[0]) + nr_ghost_cells,
            wp.int32(float_ijk[1]) + nr_ghost_cells,
            wp.int32(float_ijk[2]) + nr_ghost_cells,
        )
        return cell_index

    @wp.func
    def get_solid_fraction_stencil(
        pos: wp.vec3,
        material_properties: MaterialProperties,
    ):

        # Get center cell index
        center_cell_index = Pusher.pos_to_cell_index(
            pos,
            material_properties.id.origin,
            material_properties.id.spacing,
            material_properties.id.nr_ghost_cells,
        )
        center_cell_pos = (
            material_properties.id.origin
            + wp.cw_mul(material_properties.id.spacing,
                wp.vec3(
                    wp.float32(center_cell_index[0] - material_properties.id.nr_ghost_cells),
                    wp.float32(center_cell_index[1] - material_properties.id.nr_ghost_cells),
                    wp.float32(center_cell_index[2] - material_properties.id.nr_ghost_cells),
                )
            )
            + 0.5 * material_properties.id.spacing
        )

        # Get id for all corners of cell
        solid_fraction_stencil = Pusher.solid_fraction_stencil_type()
        solid_type_stencil = Pusher.solid_type_stencil_type()
        pos_stencil = Pusher.pos_stencil_type()
        for i in range(-2, 3):
            for j in range(-2, 3):
                for k in range(-2, 3):
                    # Get id
                    id = material_properties.id.data[
                        center_cell_index[0] + i,
                        center_cell_index[1] + j,
                        center_cell_index[2] + k,
                    ]

                    # Get solid fraction
                    sf = material_properties.solid_fraction_mapping[warp.int32(id)]

                    # Get position
                    cell_pos = (
                        center_cell_pos
                        + wp.cw_mul(
                            wp.vec3(warp.float32(i), warp.float32(j), warp.float32(k)),
                            material_properties.id.spacing
                        )
                    )

                    # Store id, solid fraction and position
                    index = (i + 2) * 25 + (j + 2) * 5 + (k + 2)
                    solid_fraction_stencil[index] = sf
                    pos_stencil[index, 0] = cell_pos[0]
                    pos_stencil[index, 1] = cell_pos[1]
                    pos_stencil[index, 2] = cell_pos[2]

        return solid_fraction_stencil, pos_stencil

    @wp.func
    def solid_fraction_stencil_to_sdf(
        pos: wp.vec3,
        solid_fraction_stencil: solid_fraction_stencil_type,
        pos_stencil: pos_stencil_type,
        spacing: wp.vec3,
    ):

        # Use sdf to get solid fraction at relative position
        sdf = spacing[0]
        for i in range(125):
            if solid_fraction_stencil[i] > 0.5:
                cell_pos = wp.vec3(pos_stencil[i, 0], pos_stencil[i, 1], pos_stencil[i, 2])
                sdf = smooth_union(sdf, wp.length(pos - cell_pos) - (spacing[0] * 1.0), spacing[0] * 1.0)
        return sdf

    @wp.func
    def solid_fraction_stencil_to_sdf_gradient(
        pos: wp.vec3,
        solid_fraction_stencil: solid_fraction_stencil_type,
        pos_stencil: pos_stencil_type,
        spacing: wp.vec3,
    ):

        # Compute normal with finite difference
        dx = spacing[0] * 1e-4

        # Compute gradient
        sdf_x_plus = Pusher.solid_fraction_sdf(
            pos + wp.vec3(dx, 0.0, 0.0),
            solid_fraction_stencil,
            pos_stencil,
            spacing,
        )
        sdf_x_minus = Pusher.solid_fraction_sdf(
            pos - wp.vec3(dx, 0.0, 0.0),
            solid_fraction_stencil,
            pos_stencil,
            spacing,
        )
        sdf_y_plus = Pusher.solid_fraction_sdf(
            pos + wp.vec3(0.0, dx, 0.0),
            solid_fraction_stencil,
            pos_stencil,
            spacing,
        )
        sdf_y_minus = Pusher.solid_fraction_sdf(
            pos - wp.vec3(0.0, dx, 0.0),
            solid_fraction_stencil,
            pos_stencil,
            spacing,
        )
        sdf_z_plus = Pusher.solid_fraction_sdf(
            pos + wp.vec3(0.0, 0.0, dx),
            solid_fraction_stencil,
            pos_stencil,
            spacing,
        )
        sdf_z_minus = Pusher.solid_fraction_sdf(
            pos - wp.vec3(0.0, 0.0, dx),
            solid_fraction_stencil,
            pos_stencil,
            spacing,
        )

        # Compute normal
        normal = wp.vec3(
            sdf_x_plus - sdf_x_minus,
            sdf_y_plus - sdf_y_minus,
            sdf_z_plus - sdf_z_minus,
        )

        # Normalize gradient
        normal = normal / wp.length(normal)

        return normal
