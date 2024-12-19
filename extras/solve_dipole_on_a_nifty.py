import argparse
import os

import numpy as np
import pandas as pd
import nibabel
import nibabel.affines
from tqdm import tqdm


def point_potential_solve_mesh(electrode_position, mesh, conductivity):
    distance_to_electrode = np.linalg.norm(np.array(electrode_position) - mesh.points, ord=2, axis=1)
    v_kcsd = 1.0 / (4 * np.pi * conductivity * distance_to_electrode)
    return v_kcsd


def main():
    # radial dipole

    a = 0.03
    mag = 100 / 1e6
    sigma = 0.33

    meshfile = "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/data/generated/tutorial/four_spheres/four_spheres_material/material.nii.gz"
    outdir = "/home/mdovgialo/test_dipole_below/analytical_dipole_free_space"

    os.makedirs(outdir, exist_ok=True)

    img = nibabel.load(meshfile)

    data = img.get_fdata()

    shape = data.shape
    voxel_coords = np.array(np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing='ij'
    )).reshape(3, -1).T

    affine = img.affine

    # mm to meters
    world_coords = nibabel.affines.apply_affine(affine, voxel_coords) / 1000



    r = np.sqrt((world_coords ** 2).sum(axis=1))


    dipole_vec = np.array([0, 0, 1])
    dipoles_pos = np.abs([0, 0, a])

    cos_theta = np.sum((world_coords / r[:, None]) * dipole_vec, axis=1)

    potential_flat = mag / (4 * np.pi * sigma) * (r - a * cos_theta) / np.sqrt(r**2 + a**2 - 2*a*r*cos_theta) ** 3
    potential_3d = potential_flat.reshape(shape)

    img = nibabel.Nifti1Image(potential_3d, affine)
    img.header.set_xyzt_units(xyz=2)  # mm
    nibabel.save(img, os.path.join(outdir, "wzor_4.nii.gz"))


    potential_flat = cos_theta
    potential_3d = potential_flat.reshape(shape)

    img = nibabel.Nifti1Image(potential_3d, affine)
    img.header.set_xyzt_units(xyz=2)  # mm
    nibabel.save(img, os.path.join(outdir, "cos_theta.nii.gz"))



    potential_flat = mag /(4 * np.pi * sigma * np.sqrt(r**2 + a**2 - 2*a*r*cos_theta))
    potential_3d = potential_flat.reshape(shape)

    img = nibabel.Nifti1Image(potential_3d, affine)
    img.header.set_xyzt_units(xyz=2)  # mm
    nibabel.save(img, os.path.join(outdir, "wzor_1.nii.gz"))


    world_coords_local = world_coords - dipoles_pos
    r_local =  np.sqrt((world_coords_local ** 2).sum(axis=1))

    cos_theta_local = np.sum((world_coords_local / r_local[:, None]) * dipole_vec, axis=1)

    potential_flat = mag * cos_theta_local / (4 * np.pi * sigma * r_local ** 2)
    potential_3d = potential_flat.reshape(shape)

    # http://www.physicsbootcamp.org/section-Electric-Potential-of-a-Dipole.html#subsection-573
    img = nibabel.Nifti1Image(potential_3d, affine)
    img.header.set_xyzt_units(xyz=2)  # mm
    nibabel.save(img, os.path.join(outdir, "wzor_klasyczny.nii.gz"))


if __name__ == '__main__':
    main()
