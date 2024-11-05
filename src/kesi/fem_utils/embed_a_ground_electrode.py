from collections import defaultdict

import gmsh
import numpy as np
import pyvista
brain_path = "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/data/generated/meshes/fsaverage/fsaverage_brain_materials_2mm.nii_volume.vtk"


gmsh.initialize()
gmsh.model.add("brain")
gmsh.merge(brain_path)
pv_mesh = pyvista.read(brain_path)
cell_data = pv_mesh.cell_data['material']

unique_material_ids = np.unique(cell_data)

for material_id in sorted(unique_material_ids):
    # Find the cell tags corresponding to the current material ID
    cell_tags = [i + 1 for i, tag in enumerate(cell_data) if tag == material_id]  # +1 to adjust for GMSH's 1-based indexing

    # Create a physical group for this material
    gmsh.model.addPhysicalGroup(3, cell_tags, name=f"Material_{material_id}")  # Change dimension if necessary
