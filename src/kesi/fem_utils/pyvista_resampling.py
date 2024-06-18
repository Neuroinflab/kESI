import argparse
import os

import nibabel
import numpy as np
import pandas as pd

import pyvista
from nibabel import Nifti1Image
from tqdm import tqdm

from kesi.fem_utils.grid_utils import create_grid, load_or_create_grid


def main():
    parser = argparse.ArgumentParser(description="samples mesh solution using vista resample")
    parser.add_argument("meshfile", help='VTK mesh file with solution')
    parser.add_argument("electrodefile",
                        help=('CSV with electrode names and positions, in meters, with a header of: \n'
                              '\tNAME,X,Y,Z')
                        )
    parser.add_argument("output", help="Output folder")
    parser.add_argument("-bc", "--base-conductivity", type=float,
                        help="base conductivity for kCSD assumptions", default=0.33)

    parser.add_argument("--attribute", help='VTK attribute to sample', default='potential')
    parser.add_argument('-s', "--sampling-step", type=float, help="step of the sampling grid", default=0.001)
    parser.add_argument('-g', "--grid-file", type=str, help="grid file, if provided sampling step is ignored", default=None)
    parser.add_argument('--nifti', dest='nifti', action='store_true',
                        help='Additionally store sampled potential in nifti format')
    parser.set_defaults(nifti=False)

    parser.add_argument('--ignore-electrode-errors', dest='ignore_electrode_errors', action='store_true',
                        help='set location to Nan on electrode error')
    parser.set_defaults(ignore_electrode_errors=False)

    namespace = parser.parse_args()
    mesh = pyvista.read(namespace.meshfile, progress_bar=True)
    electrodes = pd.read_csv(namespace.electrodefile)

    vertices = mesh.points

    grid, affine = load_or_create_grid(vertices, step=namespace.sampling_step, gridfile=namespace.grid_file)

    str_grid = pyvista.ImageData(dimensions=grid[0].shape,
                                 spacing=[affine[0][0], affine[1][1], affine[2][2]],
                                 origin=[affine[0][3], affine[1][3], affine[2][3]]
                                 )
    result = str_grid.sample(mesh, progress_bar=True, snap_to_closest_point=True)

    to_sample = [i for i in result.array_names if i.startswith(namespace.attribute)]
    save_names = []
    for i in to_sample:
        line = i.split("_", maxsplit=1)
        if len(line) > 1:
            name = line[-1]
        else:
            name = line[0]
        save_names.append(name)

    os.makedirs(namespace.output, exist_ok=True)

    for nr, save_name in enumerate(tqdm(save_names, "saving")):
        array_name = to_sample[nr]
        sampled_grid = result[array_name].reshape((*result.dimensions, 1), order="F")
        CORRECTION_POTENTIAL = sampled_grid[:, :, :, 0]
        X = grid[0][:, 0, 0][:, None, None]
        Y = grid[1][0, :, 0][None, :, None]
        Z = grid[2][0, 0, :][None, None, :]
        if not namespace.ignore_electrode_errors:
            if np.sum(electrodes.NAME == save_name) > 1:
                raise Exception("Multiple electrodes have the same names")
            if np.sum(electrodes.NAME == save_name) == 0:
                raise Exception("Electrode {} not found".format(save_name))
        try:
            LOCATION = electrodes[electrodes.NAME == save_name][['X', 'Y', 'Z']].values[0]
        except IndexError:
            LOCATION = np.array([np.nan, np.nan, np.nan])
        BASE_CONDUCTIVITY = np.array(namespace.base_conductivity)

        outfile = os.path.join(namespace.output, save_name + '.npz')
        np.savez(outfile, CORRECTION_POTENTIAL=CORRECTION_POTENTIAL,
                 X=X,
                 Y=Y,
                 Z=Z,
                 LOCATION=LOCATION,
                 BASE_CONDUCTIVITY=BASE_CONDUCTIVITY
                 )

        if namespace.nifti:
            affine_nifti = np.copy(affine)
            affine_nifti = affine_nifti * 1000
            affine_nifti[3][3] = 1
            img = Nifti1Image(sampled_grid, affine_nifti)
            img.header.set_xyzt_units(xyz=2)  # mm
            outfile = os.path.join(namespace.output, save_name + '.nii.gz')
            nibabel.save(img, outfile)


if __name__ == '__main__':
    main()