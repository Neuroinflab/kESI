import numpy as np
import pyvista as pv

conductivity = 0.33
conductivity = 1e-10
electrode_position = np.array([0,0,0])


mesh =  pv.read("/home/mdovgialo/projects/halje_data_analysis/kESI/extras/data/generated/tutorial/four_spheres/mfem_leadfield_corrections_boundary_big/four_spheres_in_air_boundary_biggest_0.005.vtk")
verts = mesh.points

distance_to_electrode = np.linalg.norm(np.array(electrode_position) - verts, ord=2, axis=1)
v_kcsd = 1.0 / (4 * np.pi * conductivity * distance_to_electrode)

mesh.point_data['potential_zero'] = v_kcsd

mesh.save("/home/mdovgialo/projects/halje_data_analysis/kESI/extras/data/generated/tutorial/four_spheres/mfem_leadfield_corrections_boundary_big/kcsd_{}_0.005.vtk".format(conductivity))
