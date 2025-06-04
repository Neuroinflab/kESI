import sys

import pyvista as pv
import numpy as np

def compare_vtk_files(file1, file2, atol=1e-8):
    mesh1 = pv.read(file1)
    mesh2 = pv.read(file2)

    # Compare number of points
    if mesh1.n_points != mesh2.n_points:
        print("Different number of points")
        return False

    # Compare point coordinates
    if not np.allclose(mesh1.points, mesh2.points, atol=atol):
        print("Point coordinates differ")
        return False

    # Compare number of cells
    if mesh1.n_cells != mesh2.n_cells:
        print("Different number of cells")
        return False

    # Compare cell connectivity
    if not np.array_equal(mesh1.cells, mesh2.cells):
        print("Cell connectivity differs")
        return False

    # Compare point data arrays
    if set(mesh1.point_data.keys()) != set(mesh2.point_data.keys()):
        print("Point data array names differ")
        return False

    for name in mesh1.point_data:
        if not np.allclose(mesh1.point_data[name], mesh2.point_data[name], atol=atol):
            print(f"Point data array '{name}' differs")
            return False

    # Compare cell data arrays
    if set(mesh1.cell_data.keys()) != set(mesh2.cell_data.keys()):
        print("Cell data array names differ")
        return False

    for name in mesh1.cell_data:
        print("Testing cell data", name)
        if not np.allclose(mesh1.cell_data[name], mesh2.cell_data[name], atol=atol):
            print(f"Cell data array '{name}' differs")
            return False

    print("VTK files are equivalent.")
    return True


if __name__ == '__main__':
    compare_vtk_files(sys.argv[-1], sys.argv[-2])