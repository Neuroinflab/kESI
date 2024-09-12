import argparse
import os.path

import nibabel
import numpy as np
import pandas as pd
import pylab as pb
from nibabel.affines import apply_affine

def create_meshgrid_from_affine(affine, data):
    x, y, z, comp = data.shape

    # Create meshgrid in image voxel coordinates
    meshgrid_x, meshgrid_y, meshgrid_z = np.mgrid[0:x, 0:y, 0:z]

    # Convert meshgrid to real space coordinates
    meshgrid_coords = np.array([meshgrid_x.ravel(), meshgrid_y.ravel(), meshgrid_z.ravel(), np.ones(meshgrid_x.size)])
    real_coords = np.dot(affine, meshgrid_coords).T
    real_coords = real_coords[:, :3].reshape(meshgrid_x.shape + (3,))
    return real_coords


def main():
    parser = argparse.ArgumentParser(description="Draw crossection nifty files, works well only with no skew or rotation, rectilinear affine")
    parser.add_argument("files", nargs='+', type=str, help="nifty files")
    parser.add_argument("-x", type=int, help="Slice at X coordinate (in meters)", default=0)
    parser.add_argument("-y", type=int, help="Slice at Y coordinate (in meters)", default=0)

    args = parser.parse_args()

    slice_of_interest = np.s_[args.x, args.y, :]

    fig = pb.figure()

    for file in args.files:
        name = os.path.basename(file)
        correction = nibabel.load(file)
        vol_data = correction.get_fdata()

        inv_affine = np.linalg.inv(correction.affine)

        x_vox, y_vox, z_vox = apply_affine(inv_affine, [args.x, args.y, 0])
        slice_of_interest = np.s_[int(x_vox), int(y_vox), :]

        meshgrid = create_meshgrid_from_affine(correction.affine, vol_data) / 1000 # mm to meters
        data_slice = vol_data[slice_of_interest]
        data_x = meshgrid[slice_of_interest]
        data_x = data_x[:, np.where((np.diff(data_x, axis=0)!=0).all(axis=0))[0][0]].squeeze()


        pb.plot(data_x, data_slice, label=name)


    pb.legend()
    pb.show()


if __name__ == '__main__':
    main()