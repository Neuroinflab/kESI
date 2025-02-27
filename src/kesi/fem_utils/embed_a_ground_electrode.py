
import numpy as np
import pyvista

# todo
# if there a big mismatch, clip is acting weirdly and unacurately
# if the electrode is smalelr than voxel, it won't cut anything using clip...

brain_path = "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/data/generated/meshes/fsaverage/fsaverage_brain_materials_2mm.nii_volume.vtk"
brain_mesh = pyvista.read(brain_path)

extra_boundaries = (0.15, 0.01, 0.5, 0.05, 10.0, 1.0, 200.0, 10.0)
extra_boundaries = (0.15, 0.01, 0.5, 0.05, 10.0, 1.0)
extra_boundaries = (0.05, 0.01, 0.10, 0.02, 0.20, 0.03)
merge_tolerance = 0.00001
electrode_coord = (0.001, -0.122, -0.005)
electrode_coord = (0.0, 0.0,0.0)
electrode_radius = 0.001

average_cell_size =  brain_mesh.get_cell(0).cast_to_unstructured_grid().compute_cell_sizes()["Volume"][0] ** (1/3)

if electrode_radius < average_cell_size:
    electrode_radius = average_cell_size
electrode = pyvista.Sphere(radius=electrode_radius, center=electrode_coord)
electrode.cell_data["material"] = np.array([10] * electrode.n_cells, dtype=brain_mesh.cell_data.active_scalars.dtype)
brain_with_electrode = brain_mesh.clip_surface(electrode, invert=False, progress_bar=True, crinkle=False)


boundary_sizes = extra_boundaries[::2]
boundary_resolutions = extra_boundaries[1::2]
boundaries = list(reversed(sorted([[i, j] for i, j in zip(boundary_sizes, boundary_resolutions)])))

boundary_meshes = []
for size, res in boundaries:
    print(size, res)
    coords = np.arange(-size, size, res)
    x, y, z = np.meshgrid(coords, coords, coords, indexing='ij')
    boundary = pyvista.UnstructuredGrid(pyvista.StructuredGrid(x, y, z))
    boundary.cell_data['material'] = np.array([np.max(np.unique(brain_with_electrode.cell_data['material']))] * boundary.n_cells, dtype = brain_with_electrode.cell_data['material'].dtype)
    boundary_meshes.append(boundary)


meshes_to_merge = boundary_meshes + [brain_with_electrode, ]
meshes_to_merge = boundary_meshes

## TODO
# this makes invalid topology!!!
final_mesh = None
for current in meshes_to_merge:
    if final_mesh is None:
        final_mesh = current
        continue
    surface = current.extract_surface()
#     clipped = final_mesh.clip_surface(surface, invert=False, progress_bar=True)
    clipped = final_mesh.clip_box(current.bounds, invert=True, progress_bar=True)
    final_mesh = clipped.merge(current, merge_points=True, tolerance=merge_tolerance, inplace=False, progress_bar=True)
# final_mesh = final_mesh.clean(tolerance=merge_tolerance, progress_bar=True)
meshes_to_merge[0].save("/home/mdovgialo/testing_electrode_embedding/weird_boundary.vtu", binary=False)
final_mesh.save("/home/mdovgialo/testing_electrode_embedding/weird_boundary.vtu", binary=False)





coords_bigger = np.arange(-200, 200, 0.5)
boundary_bigger = pyvista.StructuredGrid(coords_bigger, coords_bigger, coords_bigger)


coords_big = np.arange(-0.5, 0.5, 0.020)
boundary_big = pyvista.StructuredGrid(coords_big, coords_big, coords_big)




import pyvista

# todo
# if there a big mismatch, clip is acting weirdly and unacurately
# if the electrode is smalelr than voxel, it won't cut anything using clip...

brain_path = "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/data/generated/meshes/fsaverage/fsaverage_brain_materials_2mm.nii_volume.vtk"
brain_mesh = pyvista.read(brain_path)

clipped = brain_mesh.clip_box((-0.03, 0.03, -0.03, 0.03,-0.03, 0.03), invert=True)
clipped_tesselated = clipped.tessellate(2)
brain_mesh_clipped = brain_mesh.clip_box((-0.03, 0.03, -0.03, 0.03,-0.03, 0.03), invert=False)
merged = brain_mesh_clipped.merge(clipped_tesselated, tolerance=merge_tolerance)


