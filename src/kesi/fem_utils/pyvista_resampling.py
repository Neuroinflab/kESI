import argparse
import glob
import os
import tempfile
from io import StringIO

import nibabel
import numpy as np
import pandas as pd

import pyvista
from nibabel import Nifti1Image
from tqdm import tqdm

from kesi.fem_utils.grid_utils import load_or_create_grid
import mfem.ser as mfem

from kesi.fem_utils.vtk_utils import grid_function_save_vtk
from kesi.mfem_solver.mfem_piecewise_solver import prepare_fespace


def convert_mfem_to_pyvista(mesh, solutions, names):
    """
    mesh - mfem mesh object
    solutions - list gridfunctions
    names - list of gridfunction names to use
    """

    assert len(solutions) == len(names)
    with tempfile.NamedTemporaryFile(suffix='.vtk') as fp:
        output = StringIO()
        print("saving output mesh")
        mesh.PrintVTK(output)
        with open(fp.name, 'a') as vtk_file:
            vtk_file.write(output.getvalue())
        del output

        fespace = prepare_fespace(mesh)
        x = mfem.GridFunction(fespace)

        with open(fp.name, 'a') as vtk_file:
            vtk_file.write("POINT_DATA " + str(mesh.GetNV()) + "\n")
            for name, sol in zip(names, solutions):
                x.Assign(sol.GetDataArray())
                grid_function_save_vtk(x, vtk_file, name)

        pyvista_mesh = pyvista.read(fp.name)
    return pyvista_mesh


def pyvista_sample_points(pyvista_mesh, points):
    point_cloud = pyvista.PolyData(points)
    sampled = point_cloud.sample(pyvista_mesh, progress_bar=True, snap_to_closest_point=True)
    return sampled


def pyvista_sample_grid(pyvista_mesh, grid):
    """
    grid - meshgrid [x, y, z, 3]
    """
    dimensions = grid.shape[0:4]
    spacing = [
        grid[1, 0, 0, 0] - grid[0, 0, 0, 0],
        grid[0, 1, 0, 1] - grid[0, 0, 0, 1],
        grid[0, 0, 1, 2] - grid[0, 0, 0, 2],

    ]
    origin = [grid[0, 0, 0, 0], grid[0, 0, 0, 1], grid[0, 0, 0, 2]]

    str_grid = pyvista.ImageData(dimensions=dimensions,
                                 spacing=spacing,
                                 origin=origin
                                 )
    sampled = str_grid.sample(pyvista_mesh, progress_bar=True, snap_to_closest_point=True)
    return sampled


def main():
    parser = argparse.ArgumentParser(description="samples mesh solution using vista resample")
    parser.add_argument("meshfile",
                        help=('VTK mesh file with solutions. '
                              'If there are *.npz  files in the same folder, '
                              'solutions from those files will be used instead'))
    parser.add_argument("electrodefile",
                        help=('CSV with electrode names and positions, in meters, with a header of: \n'
                              '\tNAME,X,Y,Z')
                        )
    parser.add_argument("output", help="Output folder")
    parser.add_argument("-bc", "--base-conductivity", type=float,
                        help="base conductivity for kCSD assumptions", default=0.33)

    parser.add_argument("--attribute", help='VTK attribute to sample', default='correction')
    parser.add_argument('-s', "--sampling-step", type=float, help="step of the sampling grid", default=0.001)
    parser.add_argument('-g', "--grid-file", type=str, help="grid file, if provided sampling step is ignored",
                        default=None)
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

    array_names = list(mesh.array_names)
    npz_files = list(glob.glob(os.path.join(os.path.dirname(namespace.meshfile), '*.npz')))
    npz_file_names = [os.path.splitext(os.path.basename(i))[0] for i in npz_files]
    array_names = list(set(array_names + npz_file_names))

    to_sample = [i for i in array_names if i.startswith(namespace.attribute)]
    save_names = []
    for i in to_sample:
        line = i.split("_", maxsplit=1)
        if len(line) > 1:
            name = line[-1]
        else:
            name = line[0]
        save_names.append(name)

    os.makedirs(namespace.output, exist_ok=True)

    mesh_copy = mesh.copy()
    mesh_copy.clear_point_data()

    for nr, save_name in enumerate(tqdm(save_names, "resampling and saving")):
        array_name = to_sample[nr]
        dirname = os.path.dirname(namespace.meshfile)
        npz_file_path = os.path.join(dirname, array_name + '.npz')
        if os.path.exists(npz_file_path):
            print("Using solution from", npz_file_path, array_name)
            npz_data = np.load(npz_file_path)['sol']
            assert npz_data.shape[0] == mesh_copy.points.shape[0]
            mesh_copy.point_data[array_name] = npz_data
        else:
            print("Using solution from", namespace.meshfile, array_name)
            mesh_copy.point_data[array_name] = mesh.point_data[array_name]

        result = str_grid.sample(mesh_copy, progress_bar=True, snap_to_closest_point=True)
        mesh_copy.clear_point_data()

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
