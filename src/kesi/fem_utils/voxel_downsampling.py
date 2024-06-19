import argparse
import glob
import os.path
from functools import partial

import nibabel
import numpy as np
import pandas as pd
import pyvista
from nibabel import Nifti1Image
from sklearn.metrics import DistanceMetric
from sklearn.neighbors import KDTree
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from kesi.fem_utils.grid_utils import create_grid, load_or_create_grid


def sample_points(query_points, kdtree, values, sampling_size=0.01, empty=0.0):
    chunk_size = 10000
    chunks = int(np.ceil(query_points.shape[0] / chunk_size))
    sampled = []
    for chunk_id in tqdm(list(range(chunks)), desc='sampling by querying kdtree'):
        chunk = query_points[chunk_id * chunk_size:(chunk_id + 1) * chunk_size]
        indices_per_query_points = kdtree.query_radius(chunk, r=sampling_size / 2,)
        for indices in indices_per_query_points:
            if len(indices) > 0:
                value = np.mean(values[indices])
            else:
                value = empty
            sampled.append(value)
    return np.array(sampled)


def _sample_points(chunk, kdtree, values, sampling_size, empty):
    sampled = []
    indices_per_query_points = kdtree.query_radius(chunk, r=sampling_size / 2, )
    for indices in indices_per_query_points:
        if len(indices) > 0:
            value = np.mean(values[indices])
        else:
            value = empty
        sampled.append(value)
    return sampled


def sample_points_parallel(query_points, kdtree, values, sampling_size=0.01, empty=0.0):
    chunk_size = 10000
    chunks = int(np.ceil(query_points.shape[0] / chunk_size))

    chunks_list = []
    for chunk_id in tqdm(list(range(chunks)), desc='chunking'):
        chunk = query_points[chunk_id * chunk_size:(chunk_id + 1) * chunk_size]
        chunks_list.append(chunk)

    fn = partial(_sample_points, kdtree=kdtree, values=values, sampling_size=sampling_size, empty=empty)
    sampled = process_map(fn, chunks_list, desc='sampling by querying kdtree', chunksize=100)

    sampled = np.concatenate(sampled)
    return sampled


def voxel_downsampling(points, values, lower_bound=np.array([0, 0, 0]), upper_bound=np.array([1, 1, 1]), step=0.01,
                       empty=0.0, mesh_max_elem_size=None, sampling_metric='cityblock', grid=None, affine=None
                       ):
    """Performs voxel downsampling of a scalar field, resulting a grid of positions
    points - np.array of points of the point cloud shape: (n_points, 3)
    values np. array of n_points of the scalar field
    """

    if grid is None or affine is None:
        grid, affine = create_grid([[lower_bound[0], upper_bound[0]],
                                    [lower_bound[1], upper_bound[1]],
                                    [lower_bound[2], upper_bound[2]],
                                    ], dx=step, pad=0)
    else:
        step = np.max([affine[0][0], affine[1][1], affine[2][2]])
        print("Using step ", step)

    if mesh_max_elem_size is None:
        mesh_max_elem_size = step * 2
    if step > mesh_max_elem_size * 2:
        sampling_size = step
    else:
        sampling_size = mesh_max_elem_size * 2

    grid_points = np.array([grid[0].ravel(),
                           grid[1].ravel(),
                           grid[2].ravel(),
                           ]).T
    print("Building a kdtree...")
    kdtree = KDTree(points, metric=sampling_metric)
    print("Building a kdtree, done")

    sampled = sample_points_parallel(grid_points, kdtree, values, sampling_size=sampling_size, empty=empty)

    sampled_grid = sampled.reshape(list(grid[0].shape) + [-1, ])
    return sampled_grid, affine


def main():
    parser = argparse.ArgumentParser(description="samples mesh solution using voxel downsampling")
    parser.add_argument("meshfile",
                        help=('VTK mesh file with solution. '
                              'If there is a *_solutions.npz file in the same folder, '
                              'solutions from that file will be used instead'))
    parser.add_argument("electrodefile",
                        help=('CSV with electrode names and positions, in meters, with a header of: \n'
                              '\tNAME,X,Y,Z')
                        )
    parser.add_argument("output", help="Output folder")
    parser.add_argument("-bc", "--base-conductivity", type=float,
                        help="base conductivity for kCSD assumptions", default=0.33)

    parser.add_argument("--attribute", help='VTK attribute to sample', default='potential')
    parser.add_argument('-s', "--sampling-step", type=float, help="step of the sampling grid", default=0.001)
    parser.add_argument('-ms', "--mesh_max_elem_size", type=float, help="maximum element size of the mesh", default=0.005)  # todo make it automatic
    parser.add_argument('-g', "--grid-file", type=str, help="grid file, if provided sampling step is ignored", default=None)
    parser.add_argument('--nifti', dest='nifti', action='store_true',
                        help='Additionally store sampled potential in nifti format')

    namespace = parser.parse_args()
    mesh = pyvista.read(namespace.meshfile, progress_bar=True)
    electrodes = pd.read_csv(namespace.electrodefile)

    vertices = mesh.points

    grid, affine = load_or_create_grid(vertices, step=namespace.sampling_step, gridfile=namespace.grid_file)

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

    for nr, save_name in enumerate(tqdm(save_names, "resampling and saving")):

        array_name = to_sample[nr]
        dirname = os.path.dirname(namespace.meshfile)
        npz_file_path = os.path.join(dirname, array_name + '.npz')
        if os.path.exists(npz_file_path):
            print("Using solution from", npz_file_path, array_name)
            npz_data = np.load(npz_file_path)['sol']
            assert npz_data.shape[0] == mesh.points.shape[0]
            sol = npz_data
        else:
            print("Using solution from", namespace.meshfile, array_name)
            sol = mesh.point_data[array_name]

        metric = DistanceMetric.get_metric('minkowski', p=np.inf)
        sampled_solution, affine = voxel_downsampling(np.array(vertices), sol, grid=grid, affine=affine,
                                                      mesh_max_elem_size=namespace.mesh_max_elem_size,
                                                      sampling_metric=metric)

        sampled_grid = sampled_solution.squeeze()
        CORRECTION_POTENTIAL = sampled_grid
        X = grid[0][:, 0, 0][:, None, None]
        Y = grid[1][0, :, 0][None, :, None]
        Z = grid[2][0, 0, :][None, None, :]
        if np.sum(electrodes.NAME == save_name) > 1:
            raise Exception("Multiple electrodes have the same names")
        if np.sum(electrodes.NAME == save_name) == 0:
            raise Exception("Electrode {} not found".format(save_name))
        LOCATION = electrodes[electrodes.NAME == save_name][['X', 'Y', 'Z']].values[0]
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
            affine_nifti = affine * 1000
            affine_nifti[3][3] = 1
            img = Nifti1Image(sampled_grid, affine_nifti)
            img.header.set_xyzt_units(xyz=2)  # mm
            outfile = os.path.join(namespace.output, save_name + '.nii.gz')
            nibabel.save(img, outfile)


if __name__ == "__main__":
    main()