import argparse
import os.path
from functools import partial

import nibabel
import numpy
import numpy as np
from nibabel import Nifti1Image
from sklearn.neighbors import KDTree
from tqdm import tqdm
import mfem.ser as mfem
from tqdm.contrib.concurrent import process_map


def stacked_id(id, shape):
    z = id % shape[2]
    y = id // shape[2] % shape[1]
    x = id // (shape[2] * shape[1]) % shape[0]
    return x, y, z


def create_grid(xxyyzz, dx, pad=20):
    """

    :param xxyyzz: list of [[xmin, xmax], [ymim, ymax], [zmin, zmax]] - bounding box of the grid
    :param dx: grid spacing
    :return: meshgrid, affine
    """
    atlas_max = np.array([max(xxyyzz[0]), max(xxyyzz[1]), max(xxyyzz[2])])

    atlas_min = np.array([min(xxyyzz[0]), min(xxyyzz[1]), min(xxyyzz[2])])

    atlas_span = atlas_max - atlas_min

    atlas_max = atlas_max + atlas_span * pad / 100
    atlas_min = atlas_min - atlas_span * pad / 100

    sdx = sdy = sdz = dx

    x_real = np.arange(atlas_min[0], atlas_max[0], sdx)
    y_real = np.arange(atlas_min[1], atlas_max[1], sdy)
    z_real = np.arange(atlas_min[2], atlas_max[2], sdz)

    new_affine = np.zeros((4, 4))
    new_affine[0][0] = np.abs(sdx)
    new_affine[1][1] = np.abs(sdy)
    new_affine[2][2] = np.abs(sdz)
    # last row is always 0 0 0 1
    new_affine[3][3] = 1

    new_000 = np.array([x_real[0], y_real[0], z_real[0]])
    new_affine[0][3] = new_000[0]
    new_affine[1][3] = new_000[1]
    new_affine[2][3] = new_000[2]

    grid = np.meshgrid(x_real, y_real, z_real, indexing='ij')
    return grid, new_affine

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


def voxel_downsampling(points, values, lower_bound=np.array([0, 0 , 0]), upper_bound=np.array([1, 1, 1]), step=0.01,
                       empty=0.0, mesh_max_elem_size=None
                       ):
    """Performs voxel downsampling of a scalar field, resulting a grid of positions
    points - np.array of points of the point cloud shape: (n_points, 3)
    values np. array of n_points of the scalar field
    """

    if mesh_max_elem_size is None:
        mesh_max_elem_size = step * 2
    if step > mesh_max_elem_size * 2:
        sampling_size = step
    else:
        sampling_size = mesh_max_elem_size * 2

    grid, affine = create_grid([[lower_bound[0], upper_bound[0]],
                                [lower_bound[1], upper_bound[1]],
                                [lower_bound[2], upper_bound[2]],
                                ], dx=step, pad=0)

    grid_points = np.array([grid[0].ravel(),
                           grid[1].ravel(),
                           grid[2].ravel(),
                           ]).T
    print("Building a kdtree...")
    kdtree = KDTree(points, metric='cityblock')
    print("Building a kdtree, done")

    sampled = sample_points_parallel(grid_points, kdtree, values, sampling_size=sampling_size, empty=empty)

    sampled_grid = sampled.reshape(list(grid[0].shape) + [-1, ])
    return sampled_grid, affine


def main():
    parser = argparse.ArgumentParser(description="samples mesh solution using voxel downsampling")
    parser.add_argument("meshfile", help='MFEM mesh file .mesh')
    parser.add_argument("solutionfile", help="mfem solution file, .gf")
    parser.add_argument('-s', "--sampling-step", type=float, help="step of the sampling grid", default=0.004)

    namespace = parser.parse_args()

    mesh = mfem.Mesh(namespace.meshfile, 0, 0)
    n_elements = mesh.GetNE()
    element_sizes = [mesh.GetElementSize(i) for i in range(n_elements)]
    mesh_max_elem_size = np.max(element_sizes)

    x = mfem.GridFunction(mesh, namespace.solutionfile)

    verts = mesh.GetVertexArray()
    sol = x.GetDataArray()
    verts_n = np.array(verts)

    lower_bound = np.min(verts_n, axis=0) - np.abs(np.min(verts_n, axis=0)) * 0.5
    upper_bound = np.max(verts_n, axis=0) + np.abs(np.max(verts_n, axis=0)) * 0.5
    step = namespace.sampling_step

    sampled_solution, affine = voxel_downsampling(verts_n, sol, lower_bound=lower_bound, upper_bound=upper_bound,
                                                  step=step, mesh_max_elem_size=mesh_max_elem_size)
    img = Nifti1Image(sampled_solution, affine)
    outfile = os.path.splitext(namespace.solutionfile)[0] + '.nii.gz'
    nibabel.save(img, outfile)


if __name__ == "__main__":
    main()