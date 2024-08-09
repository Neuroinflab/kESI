import nibabel
from nibabel.affines import apply_affine
import numpy as np
import os

import argparse
import pyvista

from kesi.fem_utils.grid_utils import vertex_grid_from_volume
import vtk
from tqdm import tqdm
import tempfile
import mfem.ser as mfem
from sklearn.neighbors import KDTree


def create_cell_data_from_mri(mri, mri_grid):
    """assumes mri is in mm and mri_grid is in meters"""
    cell_centers = mri_grid.cell_centers().points * 1000
    inv_affine = np.linalg.inv(mri.affine)
    voxel_ids = apply_affine(inv_affine, cell_centers).astype(int)
    mri_data = mri.get_fdata()
    data = []
    for i in tqdm(voxel_ids, desc='sorting data'):
        data.append(mri_data[i[0], i[1], i[2]])
    return data


def main():
    parser = argparse.ArgumentParser(description=("A tool to transform partitioned MRI scan to a cube mesh "
                                                  "with materials. Assumes each voxel is marked with material index. "
                                                  "Saves MFEM compatible messh with the same name."
                                                  " Will safe mfem .mesh file with materials and grounding plate boundary defined. "
                                                  "Optionally, can save a .vtk with mesh materials and .vtu with mesh boundaries."
                                                  "For visualisation in applications which cannot load MFEM format"))
    parser.add_argument("mri",
                        help=('Segmented 3D volume file, for example .nii.gz format'))

    namespace = parser.parse_args()

    base_outfile = os.path.splitext(namespace.mri)[0]

    mri = nibabel.load(namespace.mri)
    meshgrid = vertex_grid_from_volume(mri)

    mri_grid = pyvista.StructuredGrid(meshgrid[0], meshgrid[1], meshgrid[2])
    cell_data = create_cell_data_from_mri(mri, mri_grid)

    mri_grid.cell_data['material'] = np.array(cell_data, dtype=int)

    mri_unstructured_grid = pyvista.UnstructuredGrid(mri_grid)

    surf = mri_unstructured_grid.extract_surface(progress_bar=True)
    surf.compute_normals()

    # todo hack: why not -1????
    down_cells = np.where(surf.face_normals[:, 2] == 1)[0]

    down_cell_positions = surf.cell_centers().points[surf.face_normals[:, 2] == 1]
    boundary_tree = KDTree(down_cell_positions)

    with tempfile.NamedTemporaryFile(suffix='.vtk') as fp:
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(fp.name)
        writer.SetInputData(mri_unstructured_grid)
        writer.SetFileVersion(42)
        writer.Write()

        mfem_mesh = mfem.Mesh(fp.name)
        boundary_elements_coords = []
        for i in range(mfem_mesh.GetNBE()):
            vertices = mfem_mesh.GetBdrElementVertices(i)
            coords = np.array([mfem_mesh.GetVertexArray(j) for j in vertices])
            mean_coords = coords.mean(axis=0)
            boundary_elements_coords.append(mean_coords)
        boundary_elements_coords = np.array(boundary_elements_coords)
        # todo: hack 1 mm distance
        counts = boundary_tree.query_radius(boundary_elements_coords, r=2, count_only=True)

        for i in range(mfem_mesh.GetNBE()):
            if counts[i] > 0:
                mfem_mesh.SetBdrAttribute(i, 2)
            else:
                mfem_mesh.SetBdrAttribute(i, 1)
        mfem_mesh.FinalizeMesh(0, True)
        mfem_mesh.Print(base_outfile + '.mesh')
        # todo add save vtk param
        mfem_mesh.PrintVTK(base_outfile + '.vtk')
        mfem_mesh.PrintBdrVTU(base_outfile + '.vtu')


if __name__ == '__main__':
    main()