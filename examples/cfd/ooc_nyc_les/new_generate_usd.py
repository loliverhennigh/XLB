from pxr import Usd, UsdGeom, Sdf
import os

UsdTimeCode = Usd.TimeCode

def make_valid_prim_name(name):
    # Replace problematic characters with safe ones for USD prim paths
    valid_name = name.replace('-', '_').replace('.', '_')
    return valid_name

if __name__ == '__main__':
    # Directories
    base_dir = "./output"
    boundary_dir = os.path.join(base_dir, 'boundary')
    q_criteria_dir = os.path.join(base_dir, 'q_criterion')

    # Create USD Stage
    stage = Usd.Stage.CreateNew(f"{base_dir}/output.usda")

    # Create a root Xform
    root_xform = UsdGeom.Xform.Define(stage, '/root')

    # Add all boundary objects to the root Xform
    for filename in os.listdir(boundary_dir):
        if filename.endswith('.obj'):
            obj_path = os.path.join(boundary_dir, filename)
            obj_path = os.path.abspath(obj_path)
            prim_name = os.path.splitext(filename)[0]
            prim_path = f"/root/{prim_name}"
            prim_path = make_valid_prim_name(prim_path)
            prim = stage.DefinePrim(prim_path, 'Xform')

            # Use payload instead of reference for deferred loading
            prim.GetPayloads().AddPayload(obj_path)

    # Get sorted list of q_criteria grid directories
    grid_dirs = sorted(os.listdir(q_criteria_dir))

    # Iterate over each q_criteria grid directory
    for grid_dir in grid_dirs:

        # Make valid grid directory name
        valid_grid_dir = make_valid_prim_name(grid_dir)

        # Create a new Xform for the q_criteria grid
        grid_xform = UsdGeom.Xform.Define(stage, f'/root/{valid_grid_dir}')

        # Get the q_criteria grid directory path
        grid_path = os.path.join(q_criteria_dir, grid_dir)

        # Get the list of q_criteria steps and sort them
        q_criteria_steps = sorted(os.listdir(grid_path))

        # Iterate over each q_criteria step
        for step_idx, step in enumerate(q_criteria_steps):

            # Skip non-usdc files
            if not step.endswith('.obj'):
                continue

            # Get the q_criteria step file path
            step_path = os.path.join(grid_path, step)
            step_path = os.path.abspath(step_path)

            # Make valid prim name for this step
            prim_name = make_valid_prim_name(step)
            prim_path = f"/root/{valid_grid_dir}/step_{step_idx:04d}"

            # Create a new Xform prim for each step
            step_prim = stage.DefinePrim(prim_path, 'Xform')

            # Add the payload for the object file (deferred loading)
            step_prim.GetPayloads().AddPayload(step_path)

            # Set visibility for time-sampling, visible only on the corresponding frame
            step_xform = UsdGeom.Xform(step_prim.GetPrim())
            visibility_attr = step_xform.GetVisibilityAttr()

            for frame in range(len(q_criteria_steps)):
                if frame == step_idx:
                    visibility_attr.Set(UsdGeom.Tokens.inherited, UsdTimeCode(frame))
                else:
                    visibility_attr.Set(UsdGeom.Tokens.invisible, UsdTimeCode(frame))

    # Set the time range for the USD stage based on the number of steps
    nr_steps = len(q_criteria_steps)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(nr_steps - 1)

    # Save the USD file
    stage.GetRootLayer().Save()

