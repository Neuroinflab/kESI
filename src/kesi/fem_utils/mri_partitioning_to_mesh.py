import warnings

import nibabel
from nibabel.affines import apply_affine
import numpy as np
import os

import argparse
import pyvista

from kesi.utils import write_run_summary
from kesi.fem_utils.grid_utils import vertex_grid_from_volume
from tqdm import tqdm
from nibabel.processing import conform


def create_cell_data_from_mri(mri, mri_grid, default_value):
    """assumes mri is in mm and mri_grid is in meters"""
    cell_centers = mri_grid.cell_centers().points * 1000
    inv_affine = np.linalg.inv(mri.affine)
    voxel_ids = apply_affine(inv_affine, cell_centers).round().astype(int)
    mri_data = mri.get_fdata()
    data = []
    for i in tqdm(voxel_ids, desc='assigning materials to cells'):
        try:
            if i[0] < 0 or i[1] < 0 or i[2] < 0:
                data.append(default_value)
            else:
                data.append(mri_data[i[0], i[1], i[2]])
        except IndexError:
            data.append(default_value)
    return data


def align_mri_volume_to_ras(mri):
    voxel_sizes = tuple(np.abs([mri.header.get_base_affine()[0][0],
                                mri.header.get_base_affine()[1][1],
                                mri.header.get_base_affine()[2][2],
                                ]))
    if len(mri.shape) > 3:
        mri = nibabel.funcs.squeeze_image(mri)
    mri = conform(mri, out_shape=mri.shape, voxel_size=voxel_sizes, order=0)
    return mri


def main():
    parser = argparse.ArgumentParser(description=("A tool to transform partitioned MRI scan to a cube mesh "
                                                  "with materials. Assumes each voxel is marked with material index. "
                                                  "Saves MFEM compatible messh with the same name."
                                                  ""))
    parser.add_argument("mri",
                        help=('Segmented 3D volume file, for example .nii.gz format'))
    parser.add_argument("-o", "--outdir",
                        help='output directory')
    parser.add_argument("-b", "--boundaries", nargs='+', type=float,
                        help=("extra boundaries consisting of last material,"
                              " list of size and resolution. "
                              "For  example: 0.15 0.01 0.5 0.05. For"
                              " boundaries with decreasing spatial resolution"),
                        # default=(0.5, 0.3, 200.0, 50.0))
                        default=(0.5, 0.2, 10.0, 3.0, 200.0, 10.0))
    # default=(0.15, 0.03, 0.5, 0.2, 10.0, 3.0,))
    parser.add_argument("-e", "--electrode", nargs=3, type=float,
                        help=("grounding electrode position if not given there will be only far boundary condition"),
                        default=None)
    parser.add_argument("-r", "--electrode-radius", type=float,
                        help=("Grounding electrode radius"),
                        default=0.002)
    parser.add_argument('--crinkle', action=argparse.BooleanOptionalAction,
                        help=("Instead of carving out the electrode"
                              " shape in the mesh, delete a few mesh cells")
                        )

    namespace = parser.parse_args()
    os.makedirs(namespace.outdir, exist_ok=True)
    write_run_summary(namespace.outdir, namespace)

    # extra_boundaries = (0.15, 0.01, 0.5, 0.05, 10.0, 1.0, 200.0, 10.0)
    # extra_boundaries = (0.15, 0.01, 0.5, 0.05, 10.0, 1.0, 30, 10.0)
    extra_boundaries = namespace.boundaries
    electrode_radius = namespace.electrode_radius
    electrode_position = namespace.electrode

    assert (len(extra_boundaries) % 2) == 0
    # # extra_boundaries = (0.05, 0.01, 0.10, 0.02, 0.20, 0.03)
    # merge_tolerance = 0.00001
    # electrode_coord = (0.001, -0.12, -0.005)
    # electrode_coord = (0.0, 0.0, 0.0)
    # electrode_radius = 0.003

    base_outfile = os.path.splitext(namespace.mri)[0]

    mri = nibabel.load(namespace.mri)
    # needed for good vertex orientation
    mri = align_mri_volume_to_ras(mri)

    meshgrid = vertex_grid_from_volume(mri)

    mri_grid = pyvista.StructuredGrid(meshgrid[0], meshgrid[1], meshgrid[2])

    mri_x = meshgrid[0][:, 0, 0]
    mri_y = meshgrid[1][0, :, 0]
    mri_z = meshgrid[2][0, 0, :]

    boundary_sizes = extra_boundaries[::2]
    boundary_resolutions = extra_boundaries[1::2]
    boundaries = list(sorted([[i, j] for i, j in zip(boundary_sizes, boundary_resolutions)]))

    boundary_coords = [mri_x, mri_y, mri_z]
    for b in boundaries:
        b_x = np.arange(-b[0] / 2, b[0] / 2 + b[1], b[1])
        b_x = b_x[np.logical_or(b_x < np.min(boundary_coords[0]), b_x > np.max(boundary_coords[0]))]
        final_x = np.array(sorted(list(boundary_coords[0]) + list(b_x)))

        b_y = np.arange(-b[0] / 2, b[0] / 2 + b[1], b[1])
        b_y = b_y[np.logical_or(b_y < np.min(boundary_coords[1]), b_y > np.max(boundary_coords[1]))]
        final_y = np.array(sorted(list(boundary_coords[1]) + list(b_y)))

        b_z = np.arange(-b[0] / 2, b[0] / 2 + b[1], b[1])
        b_z = b_z[np.logical_or(b_z < np.min(boundary_coords[2]), b_z > np.max(boundary_coords[2]))]
        final_z = np.array(sorted(list(boundary_coords[2]) + list(b_z)))
        boundary_coords = [final_x, final_y, final_z]

    mri_with_boundaries = pyvista.RectilinearGrid(*boundary_coords)
    mri_with_boundaries = mri_with_boundaries.cast_to_unstructured_grid()

    # default value - last material
    cell_data = create_cell_data_from_mri(mri, mri_with_boundaries,
                                          list(reversed(sorted(np.unique(mri.get_fdata()))))[0]
                                          )

    mri_with_boundaries.cell_data['material'] = np.array(cell_data, dtype=int)

    average_cell_size = mri_grid.get_cell(0).cast_to_unstructured_grid().compute_cell_sizes()["Volume"][0] ** (
            1 / 3)

    if electrode_position is not None:

        if electrode_radius < average_cell_size:
            warnings.warn("Electrode radius {} too small. Increasing electrode radius to {}".format(electrode_radius,
                                                                                                    average_cell_size))
            electrode_radius = average_cell_size

        electrode = pyvista.Sphere(radius=electrode_radius, center=electrode_position)
        electrode.cell_data["material"] = np.array([10] * electrode.n_cells,
                                                   dtype=mri_with_boundaries.cell_data.active_scalars.dtype)
        mri_with_boundaries_clipped = mri_with_boundaries.clip_surface(electrode, invert=False, progress_bar=True,
                                                                       crinkle=namespace.crinkle)

        if (mri_with_boundaries.n_cells == mri_with_boundaries_clipped.n_cells):
            raise ValueError("Radius still too small, nothing was clipped!!!")
        mri_with_boundaries = mri_with_boundaries_clipped

    else:
        # TODO: HACKS without CLIPPPING MFEM REFUSES TO LOAD THE MESH WTF
        # so we chip a corner tiny bit
        electrode = pyvista.Sphere(radius=average_cell_size,
                                   center=(mri_with_boundaries.bounds[0],
                                           mri_with_boundaries.bounds[2],
                                           mri_with_boundaries.bounds[4])
                                   )
        electrode.cell_data["material"] = np.array([10] * electrode.n_cells,
                                                   dtype=mri_with_boundaries.cell_data.active_scalars.dtype)
        mri_with_boundaries = mri_with_boundaries.clip_surface(electrode, invert=False, progress_bar=True,
                                                               crinkle=namespace.crinkle)

        # mri_with_boundaries = mri_with_boundaries.clean(progress_bar=True)
    mri_with_boundaries.save(os.path.join(namespace.outdir, os.path.basename(base_outfile) + '_volume.vtu'),
                             binary=True)


if __name__ == '__main__':
    main()
