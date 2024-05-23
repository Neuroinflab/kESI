import argparse
import os

import mfem.ser as mfem
import nibabel
import numpy as np

import pyvista
from nibabel import Nifti1Image

try:
    from .voxel_downsampling import create_grid
except ImportError:
    from voxel_downsampling import create_grid


def main():
    parser = argparse.ArgumentParser(description="samples mesh solution using voxel downsampling")
    parser.add_argument("meshfile", help='mesh with solution as .vtk')
    parser.add_argument('-s', "--sampling-step", type=float, help="step of the sampling grid", default=0.001)

    namespace = parser.parse_args()

    mesh = pyvista.read(namespace.meshfile, progress_bar=True)

    verts_n = np.array(mesh.points)
    lower_bound = np.min(verts_n, axis=0)
    upper_bound = np.max(verts_n, axis=0)

    grid, affine = create_grid([[lower_bound[0], upper_bound[0]],
                                [lower_bound[1], upper_bound[1]],
                                [lower_bound[2], upper_bound[2]],
                                ], dx=namespace.sampling_step, pad=0)

    str_grid = pyvista.ImageData(dimensions=grid[0].shape,
                      spacing=[affine[0][0], affine[1][1], affine[2][2]],
                      origin=[affine[0][3], affine[1][3], affine[2][3]]
                      )

    result = str_grid.sample(mesh, progress_bar=True, snap_to_closest_point=True)

    sampled_grid = result['potential'].reshape((*result.dimensions, 1), order="F")

    img = Nifti1Image(sampled_grid, affine)
    outfile = os.path.splitext(namespace.meshfile)[0] + '_vista.nii.gz'
    nibabel.save(img, outfile)

    # # result = grid.sample(mesh, progress_bar=True)
    # # result = grid.interpolate(mesh, progress_bar=True)
    #
    #
    # mesh = mfem.Mesh(namespace.meshfile, 0, 0)
    # n_elements = mesh.GetNE()
    # element_sizes = [mesh.GetElementSize(i) for i in range(n_elements)]
    # mesh_max_elem_size = np.max(element_sizes)
    #
    # x = mfem.GridFunction(mesh, namespace.solutionfile)
    #
    # sampled_grid = sampled.reshape(list(grid[0].shape) + [-1, ])






if __name__ == '__main__':
    main()