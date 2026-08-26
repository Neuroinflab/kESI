import os
import nibabel
import numpy as np
import pandas as pd
from nibabel.affines import apply_affine
from numpy.linalg import inv

dirname = os.path.dirname(__file__)

electrodes_path = os.path.join(dirname, '..', 'electrode_locations', '10_20', '10_20_FOUR_SPHERES_SCALP_0.089.csv')

materials = nibabel.load(os.path.join(dirname, "fsaverage_brain_materials_1mm.nii.gz"))

affine = materials.affine
affine_inv = inv(affine)

electrodes = pd.read_csv(electrodes_path)

positions = electrodes[["X", "Y", "Z"]].values
norm = np.linalg.norm(positions, axis=1)
positions_normalized = positions / norm[:, None]

material_data = materials.get_fdata()

electrode_positions_on_the_scalp = []

for electrode_id in range(len(electrodes)):

    for distance in np.arange(600, 0.0, -0.1): # in milimiters, because MRI affine is in milimiters
        x, y, z = positions_normalized[electrode_id] * distance
        x_id, y_id, z_id = apply_affine(affine_inv, [[x, y, z]]).astype(int)[0]
        if x_id < 0 or y_id < 0 or z_id < 0:
            continue
        try:
            material = material_data[x_id, y_id, z_id]
        except IndexError:
            continue

        # we are moving from far away to the head,
        # as soon as we touch the scalp, we want that electrode position
        if material == 4:  # scalp material
            electrode_position_on_the_scalp = apply_affine(affine, [[x_id, y_id, z_id]])[0] # in the middle of the voxel
            electrode_positions_on_the_scalp.append(electrode_position_on_the_scalp)
            break

electrode_positions_on_the_scalp = np.array(electrode_positions_on_the_scalp)

# to SI units (meters)
electrodes["X"] = electrode_positions_on_the_scalp[:, 0] / 1000
electrodes["Y"] = electrode_positions_on_the_scalp[:, 1] / 1000
electrodes["Z"] = electrode_positions_on_the_scalp[:, 2] / 1000

electrodes.to_csv(os.path.join(dirname, '..', 'electrode_locations', '10_20', "10_20_FSAVERAGE_SCALP.csv"), index=None)



