import argparse
import os.path

import nibabel
import numpy as np
from nibabel import Nifti1Image


def main():

    parser = argparse.ArgumentParser(description="converts .npz sampled solution file to nifti")
    parser.add_argument("file")

    namespace = parser.parse_args()

    data = np.load(namespace.file)
    dx = np.diff(data['X'].flatten())[0]
    dy = np.diff(data['Y'].flatten())[0]
    dz = np.diff(data['Z'].flatten())[0]
    potential_grid = data['CORRECTION_POTENTIAL']

    X = np.ones_like(potential_grid) * data['X']
    Y = np.ones_like(potential_grid) * data['Y']
    Z = np.ones_like(potential_grid) * data['Z']

    grid_points = np.array([X.ravel(),
                            Y.ravel(),
                            Z.ravel(),
                            ]).T

    electrode_position = [0, 0, 0.0785]
    sigma_base = 0.33
    distance_to_electrode = np.linalg.norm(np.array(electrode_position) - grid_points, ord=2, axis=1)
    v_kcsd = 1.0 / (4 * np.pi * sigma_base * distance_to_electrode)
    v_kcsd_grid = v_kcsd.reshape(list(X.shape) + [-1, ]).squeeze()

    x0 = data['X'].flatten()[0]
    y0 = data['Y'].flatten()[0]
    z0 = data['Z'].flatten()[0]
    affine = np.zeros((4, 4))
    affine[0][0] = dx
    affine[1][1] = dy
    affine[2][2] = dz
    affine[3][3] = 1.0
    affine[0][3] = x0
    affine[1][3] = y0
    affine[2][3] = z0

    img = Nifti1Image(potential_grid, affine)
    nibabel.save(img, "{}.nii.gz".format(os.path.splitext(namespace.file)[0]))

    img = Nifti1Image(potential_grid+v_kcsd_grid, affine)
    nibabel.save(img, "{}+potential.nii.gz".format(os.path.splitext(namespace.file)[0]))

    img = Nifti1Image(v_kcsd_grid, affine)
    nibabel.save(img, "{}_potential_only.nii.gz".format(os.path.splitext(namespace.file)[0]))




if __name__ == '__main__':
    main()