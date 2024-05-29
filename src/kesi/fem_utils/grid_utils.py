import numpy as np


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


def load_or_create_grid(voxels, step, gridfile=None):
    pass


def save_sampled_solution():
    pass
