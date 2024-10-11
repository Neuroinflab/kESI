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
from nibabel.processing import conform

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

def align_mri_volume_to_ras(mri):
    voxel_sizes = tuple(np.abs([mri.header.get_base_affine()[0][0],
                   mri.header.get_base_affine()[1][1],
                   mri.header.get_base_affine()[2][2],
                   ]))
    mri = conform(mri, out_shape=mri.shape, voxel_size=voxel_sizes , order=0)
    return mri

def main():
    parser = argparse.ArgumentParser(description=("A tool to transform partitioned MRI scan to a cube mesh "
                                                  "with materials. Assumes each voxel is marked with material index. "
                                                  "Saves MFEM compatible messh with the same name."
                                                  " Will safe mfem .mesh file with materials and grounding plate boundary defined. "
                                                  "Optionally, can save a .vtk with mesh materials and .vtu with mesh boundaries."
                                                  "For visualisation in applications which cannot load MFEM format"))
    parser.add_argument("mri",
                        help=('Segmented 3D volume file, for example .nii.gz format'))
    parser.add_argument("-o" "--outdir",
                        help='output directory')

    namespace = parser.parse_args()

    base_outfile = os.path.splitext(namespace.mri)[0]

    mri = nibabel.load(namespace.mri)
    # needed for good vertex orientation
    mri = align_mri_volume_to_ras(mri)

    meshgrid = vertex_grid_from_volume(mri)

    mri_grid = pyvista.StructuredGrid(meshgrid[0], meshgrid[1], meshgrid[2])
    cell_data = create_cell_data_from_mri(mri, mri_grid)

    mri_grid.cell_data['material'] = np.array(cell_data, dtype=int)

    mri_unstructured_grid = pyvista.UnstructuredGrid(mri_grid)

    surf = mri_unstructured_grid.extract_surface(progress_bar=True)
    surf.compute_normals()

    down_cell_positions = surf.cell_centers().points[surf.face_normals[:, 2] == -1]
    boundary_tree = KDTree(down_cell_positions)

    with tempfile.NamedTemporaryFile(suffix='.vtk') as fp:
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(fp.name)
        writer.SetInputData(mri_unstructured_grid)
        writer.SetFileVersion(42)
        writer.Write()

        mfem_mesh = mfem.Mesh(fp.name)
        boundary_elements_coords = []

        boundary_vertices = []
        for i in range(mfem_mesh.GetNBE()):
            vertices = mfem_mesh.GetBdrElementVertices(i)
            # todo look how boundary vertices look like, what are they, can I mark any as boundary?
            coords = np.array([mfem_mesh.GetVertexArray(j) for j in vertices])
            mean_coords = coords.mean(axis=0)
            boundary_elements_coords.append(mean_coords)
        boundary_elements_coords = np.array(boundary_elements_coords)

        import IPython
        IPython.embed()

        vox_size = np.mean(np.abs([mri.header.get_base_affine()[0][0],
                        mri.header.get_base_affine()[1][1],
                        mri.header.get_base_affine()[2][2],
                               ]
                              )
                       ) / 1000 # in meters
        counts = boundary_tree.query_radius(boundary_elements_coords, r=vox_size * 1.5, count_only=True)

        for i in range(mfem_mesh.GetNBE()):
            if counts[i] > 0:
                mfem_mesh.SetBdrAttribute(i, 2)
            else:
                mfem_mesh.SetBdrAttribute(i, 1)
        mfem_mesh.FinalizeMesh(0, True)
        mfem_mesh.Print(base_outfile + '.mesh')
        # todo add save vtk param
        mfem_mesh.PrintVTK(base_outfile + '_volume.vtk')
        mfem_mesh.PrintBdrVTU(base_outfile + '_boundaries')


if __name__ == '__main__':
    main()