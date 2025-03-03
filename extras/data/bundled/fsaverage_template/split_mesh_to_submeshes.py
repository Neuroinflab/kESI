import pyvista
import numpy as np

brain_path = "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/data/generated/meshes/fsaverage/fsaverage_brain_materials_1mm.nii_volume.vtk"

brain_mesh = pyvista.read(brain_path)
material_values = np.unique(brain_mesh.cell_data["material"])
submeshes = {m: brain_mesh.threshold(value=[m-0.01, m+0.01], scalars="material") for m in material_values}

for i in material_values:
    submeshes[i].save("/home/mdovgialo/testing_electrode_embedding/brain_submesh_{}.vtk".format(i))