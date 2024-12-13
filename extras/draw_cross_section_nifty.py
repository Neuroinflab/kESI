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
    parser.add_argument("-x", type=float, help="Slice at X coordinate (in meters)", default=0)
    parser.add_argument("-y", type=float, help="Slice at Y coordinate (in meters)", default=0)
    parser.add_argument("-g", type=float, help="position on Z axis to use as common reference, by default none", default=None)
    parser.add_argument("-f", "--frame_number", type=int, help="frame/component number", default=0)

    args = parser.parse_args()

    slice_of_interest = np.s_[args.x, args.y, :]

    fig = pb.figure()

    for file in args.files:
        name = "{} {}".format(os.path.basename(os.path.dirname(file)), os.path.basename(file))
        correction = nibabel.load(file)
        vol_data = correction.get_fdata()
        # in case Nifty has components, grab the first one
        if len(vol_data.shape) == 5:
            try:
                vol_data = correction.get_fdata()[:, : ,:, :, args.frame_number]
            except IndexError:
                vol_data = correction.get_fdata()[:, :, :, args.frame_number, :]


        inv_affine = np.linalg.inv(correction.affine)

        x_vox, y_vox, z_vox = apply_affine(inv_affine, [args.x * 1000, args.y * 1000, 0])
        slice_of_interest = np.s_[int(x_vox), int(y_vox), :]

        meshgrid = create_meshgrid_from_affine(correction.affine, vol_data) / 1000 # mm to meters
        data_slice = vol_data[slice_of_interest]
        data_x = meshgrid[slice_of_interest]
        data_x = data_x[:, np.where((np.diff(data_x, axis=0)!=0).all(axis=0))[0][0]].squeeze()

        if args.g is not None:
            ref_level_id = np.argmin(np.abs(data_x - args.g))
            ref_level = data_slice[ref_level_id]
            pb.plot(data_x, data_slice-ref_level, label=name)
        else:
            pb.plot(data_x, data_slice, label=name)

    ###############3
    a = 0.030

    x = [0, a * 2]
    p = 100 * 1e-6
    y = [-p/(4 * np.pi * 0.33 * a**2), p/(4 * np.pi * 0.33 * a**2)]
    pb.scatter(x, y, label='dipole at 0.03 analytical free space solution')


    pb.legend()
    pb.show()


if __name__ == '__main__':
    main()