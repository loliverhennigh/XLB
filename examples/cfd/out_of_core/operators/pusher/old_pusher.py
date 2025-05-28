# Base pusher class for pushing particles in time

import warp as wp

from pumpkin_pulse.struct.particles import Particles, Particle
from pumpkin_pulse.struct.material_properties import MaterialProperties
from pumpkin_pulse.operator.operator import Operator
from pumpkin_pulse.operator.sort.sort import ParticleSorter

class Pusher(Operator):
    sort_particles = ParticleSorter()

    # Define helpfull data types
    id_corners_type = wp.vec(8, dtype=wp.uint8) # id for the 8 corners of a cell (000, 100, 010, 110, 001, 101, 011, 111)
    sf_corners_type = wp.vec(8, dtype=wp.float32) # solid fraction for the 8 corners of a cell (000, 100, 010, 110, 001, 101, 011, 111)
    st_corners_type = wp.vec(8, dtype=wp.uint8) # solid type for the 8 corners of a cell (000, 100, 010, 110, 001, 101, 011, 111)

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
    def min_timestep(
        pos: wp.vec3f,
        vel: wp.vec3f,
        dt: wp.float32,
        origin: wp.vec3f,
        spacing: wp.vec3f,
    ):
        """
        Calculate the minimum time step to move particle
        """

        # Check if particle hits each wall
        # X-direction
        if vel[0] > 0.0:
            wall_pos_x = wp.float32(wp.int32((pos[0] - origin[0]) / spacing[0]) + 1) * spacing[0] + origin[0]
            dt_x = (wall_pos_x - pos[0]) / vel[0]
        elif vel[0] < 0.0:
            wall_pos_x = wp.float32(wp.int32((pos[0] - origin[0]) / spacing[0])) * spacing[0] + origin[0]
            dt_x = (wall_pos_x - pos[0]) / vel[0]
        else:
            dt_x = wp.float32(1e10)

        # Y-direction
        if vel[1] > 0.0:
            wall_pos_y = wp.float32(wp.int32((pos[1] - origin[1]) / spacing[1]) + 1) * spacing[1] + origin[1]
            dt_y = (wall_pos_y - pos[1]) / vel[1]
        elif vel[1] < 0.0:
            wall_pos_y = wp.float32(wp.int32((pos[1] - origin[1]) / spacing[1])) * spacing[1] + origin[1]
            dt_y = (wall_pos_y - pos[1]) / vel[1]
        else:
            dt_y = wp.float32(1e10)

        # Z-direction
        if vel[2] > 0.0:
            wall_pos_z = wp.float32(wp.int32((pos[2] - origin[2]) / spacing[2]) + 1) * spacing[2] + origin[2]
            dt_z = (wall_pos_z - pos[2]) / vel[2]
        elif vel[2] < 0.0:
            wall_pos_z = wp.float32(wp.int32((pos[2] - origin[2]) / spacing[2])) * spacing[2] + origin[2]
            dt_z = (wall_pos_z - pos[2]) / vel[2]
        else:
            dt_z = wp.float32(1e10)

        # Add small epsilon to get over wall
        dt_x += spacing[0] / (wp.abs(vel[0]) * 1000.0)
        dt_y += spacing[1] / (wp.abs(vel[1]) * 1000.0)
        dt_z += spacing[2] / (wp.abs(vel[2]) * 1000.0)

        # Get minimum time step
        min_dt = wp.min(wp.min(wp.min(dt_x, dt_y), dt_z), dt)

        return min_dt

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
    def pos_to_lower_cell_index(
        pos: wp.vec3,
        origin: wp.vec3,
        spacing: wp.vec3,
        nr_ghost_cells: wp.int32,
    ):
        float_ijk = wp.cw_div(pos - origin, spacing)
        lower_cell_index = wp.vec3i(
            wp.int32(float_ijk[0] - 0.5) + nr_ghost_cells,
            wp.int32(float_ijk[1] - 0.5) + nr_ghost_cells,
            wp.int32(float_ijk[2] - 0.5) + nr_ghost_cells,
        )
        return lower_cell_index

    @wp.func
    def pos_to_relative_pos(
        pos: wp.vec3,
        origin: wp.vec3,
        spacing: wp.vec3,
    ):
        relative_pos = wp.cw_div(pos - origin, spacing)
        return wp.vec3(
            relative_pos[0] - wp.float32(wp.int32(relative_pos[0])),
            relative_pos[1] - wp.float32(wp.int32(relative_pos[1])),
            relative_pos[2] - wp.float32(wp.int32(relative_pos[2])),
        )

    @wp.func
    def get_id_corners(
        lower_cell_index: wp.vec3i,
        material_properties: MaterialProperties,
    ):
        # Get id for all corners of cell
        id_corners = Pusher.id_corners_type(
            material_properties.id.data[lower_cell_index[0], lower_cell_index[1], lower_cell_index[2]],
            material_properties.id.data[lower_cell_index[0] + 1, lower_cell_index[1], lower_cell_index[2]],
            material_properties.id.data[lower_cell_index[0], lower_cell_index[1] + 1, lower_cell_index[2]],
            material_properties.id.data[lower_cell_index[0] + 1, lower_cell_index[1] + 1, lower_cell_index[2]],
            material_properties.id.data[lower_cell_index[0], lower_cell_index[1], lower_cell_index[2] + 1],
            material_properties.id.data[lower_cell_index[0] + 1, lower_cell_index[1], lower_cell_index[2] + 1],
            material_properties.id.data[lower_cell_index[0], lower_cell_index[1] + 1, lower_cell_index[2] + 1],
            material_properties.id.data[lower_cell_index[0] + 1, lower_cell_index[1] + 1, lower_cell_index[2] + 1],
        )
        return id_corners

    @wp.func
    def get_solid_fraction_corners(
        id_corners: id_corners_type,
        material_properties: MaterialProperties,
    ):
        # Get solid fraction for all corners of cell
        sf_corners = Pusher.sf_corners_type(
            material_properties.solid_fraction_mapping[wp.int32(id_corners[0])],
            material_properties.solid_fraction_mapping[wp.int32(id_corners[1])],
            material_properties.solid_fraction_mapping[wp.int32(id_corners[2])],
            material_properties.solid_fraction_mapping[wp.int32(id_corners[3])],
            material_properties.solid_fraction_mapping[wp.int32(id_corners[4])],
            material_properties.solid_fraction_mapping[wp.int32(id_corners[5])],
            material_properties.solid_fraction_mapping[wp.int32(id_corners[6])],
            material_properties.solid_fraction_mapping[wp.int32(id_corners[7])],
        )
        return sf_corners

    @wp.func
    def interpolate_solid_fraction_to_sdf(
        sf_corners: sf_corners_type,
        relative_pos: wp.vec3,
    ):
        # Use sdf to get solid fraction at relative position
        sdf = 0.25
        if sf_corners[0] > 0.5:
            sdf = wp.min(sdf, wp.length(relative_pos - wp.vec3(0.0, 0.0, 0.0)) - 1.0)
        if sf_corners[1] > 0.5:
            sdf = wp.min(sdf, wp.length(relative_pos - wp.vec3(1.0, 0.0, 0.0)) - 1.0)
        if sf_corners[2] > 0.5:
            sdf = wp.min(sdf, wp.length(relative_pos - wp.vec3(0.0, 1.0, 0.0)) - 1.0)
        if sf_corners[3] > 0.5:
            sdf = wp.min(sdf, wp.length(relative_pos - wp.vec3(1.0, 1.0, 0.0)) - 1.0)
        if sf_corners[4] > 0.5:
            sdf = wp.min(sdf, wp.length(relative_pos - wp.vec3(0.0, 0.0, 1.0)) - 1.0)
        if sf_corners[5] > 0.5:
            sdf = wp.min(sdf, wp.length(relative_pos - wp.vec3(1.0, 0.0, 1.0)) - 1.0)
        if sf_corners[6] > 0.5:
            sdf = wp.min(sdf, wp.length(relative_pos - wp.vec3(0.0, 1.0, 1.0)) - 1.0)
        if sf_corners[7] > 0.5:
            sdf = wp.min(sdf, wp.length(relative_pos - wp.vec3(1.0, 1.0, 1.0)) - 1.0)

        return sdf


    #@wp.func
    #def interpolate_solid_fraction_to_pos(
    #    sf_corners: sf_corners_type,
    #    relative_pos: wp.vec3,
    #):
    #    # Use trilinear interpolation to get solid fraction at relative position

    #    # sf_corners: 000, 100, 010, 110, 001, 101, 011, 111

    #    # x-direction
    #    f_00 = sf_corners[0] * (1.0 - relative_pos[0]) + sf_corners[1] * relative_pos[0]
    #    f_01 = sf_corners[4] * (1.0 - relative_pos[0]) + sf_corners[5] * relative_pos[0]
    #    f_10 = sf_corners[2] * (1.0 - relative_pos[0]) + sf_corners[3] * relative_pos[0]
    #    f_11 = sf_corners[6] * (1.0 - relative_pos[0]) + sf_corners[7] * relative_pos[0]

    #    # y-direction
    #    f_0 = f_00 * (1.0 - relative_pos[1]) + f_10 * relative_pos[1]
    #    f_1 = f_01 * (1.0 - relative_pos[1]) + f_11 * relative_pos[1]
    #    
    #    # z-direction
    #    f = f_0 * (1.0 - relative_pos[2]) + f_1 * relative_pos[2]

    #    return f

    @wp.func
    def interpolate_normal_solid_fraction_to_pos(
        sf_corners: sf_corners_type,
    ):
        # Uses solid fraction at corners to calculate normal solid fraction

        # Compute derivatives of solid fraction in x, y and z direction
        dsf_dx = (- sf_corners[0] + sf_corners[1] - sf_corners[2] + sf_corners[3] - sf_corners[4] + sf_corners[5] - sf_corners[6] + sf_corners[7]) / 2.0
        dsf_dy = (- sf_corners[0] - sf_corners[1] + sf_corners[2] + sf_corners[3] - sf_corners[4] - sf_corners[5] + sf_corners[6] + sf_corners[7]) / 2.0
        dsf_dz = (- sf_corners[0] - sf_corners[1] - sf_corners[2] - sf_corners[3] + sf_corners[4] + sf_corners[5] + sf_corners[6] + sf_corners[7]) / 2.0

        # Compute norm of gradient
        norm = wp.sqrt(dsf_dx * dsf_dx + dsf_dy * dsf_dy + dsf_dz * dsf_dz)

        # Compute normal
        #normal = wp.vec3(dsf_dx, dsf_dy, dsf_dz) / norm
        normal = wp.vec3(-1.0, 0.0, 0.0)

        return normal

    @wp.func
    def interpolate_solid_type_to_pos(
        id_corners: id_corners_type,
        material_properties: MaterialProperties,
        relative_pos: wp.vec3,
    ):
 
        # Nearest neighbour interpolation
        if relative_pos[0] < 0.5:
            if relative_pos[1] < 0.5:
                if relative_pos[2] < 0.5:
                    return material_properties.solid_type_mapping[wp.int32(id_corners[0])]
                else:
                    return material_properties.solid_type_mapping[wp.int32(id_corners[4])]
            else:
                if relative_pos[2] < 0.5:
                    return material_properties.solid_type_mapping[wp.int32(id_corners[2])]
                else:
                    return material_properties.solid_type_mapping[wp.int32(id_corners[6])]
        else:
            if relative_pos[1] < 0.5:
                if relative_pos[2] < 0.5:
                    return material_properties.solid_type_mapping[wp.int32(id_corners[1])]
                else:
                    return material_properties.solid_type_mapping[wp.int32(id_corners[5])]
            else:
                if relative_pos[2] < 0.5:
                    return material_properties.solid_type_mapping[wp.int32(id_corners[3])]
                else:
                    return material_properties.solid_type_mapping[wp.int32(id_corners[7])]
