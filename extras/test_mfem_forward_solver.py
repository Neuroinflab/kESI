import os.path

import IPython.utils.path
import numpy as np
import pandas as pd

from kesi.common import GaussianSourceKCSD3D
from kesi.mfem_solver.forward_solver import CSDForwardSolver

this_folder = os.path.dirname(__file__)


meshfile = os.path.join(this_folder, "data", "generated", "meshes", "four_spheres_with_electrode_0.005.msh")
meshfile = os.path.join(this_folder, "data", "generated", "meshes", "four_spheres_in_air_with_plane_0.005.msh")
conductivities = np.array([0.33, 1.65, 0.0165, 0.33, 1e-10])
# conductivities = np.array([0.33, 1.65, 0.0165, 0.33, 1e-99])

electrodes_file = os.path.join(this_folder, "data", "generated", "tutorial", "four_spheres", "tutorial_electrodes_four.csv")
sampling_points = pd.read_csv(electrodes_file)[["X", "Y", "Z"]].values

solver = CSDForwardSolver(meshfile, conductivities)

mesh_points = np.array(solver.mesh.GetVertexArray())

min_x = np.min(mesh_points[:, 0])
max_x = np.max(mesh_points[:, 0])
min_y = np.min(mesh_points[:, 1])
max_y = np.max(mesh_points[:, 1])
min_z = np.min(mesh_points[:, 2])
max_z = np.max(mesh_points[:, 2])

dx = 0.001

x = np.arange(min_x, max_x, dx)
y = np.arange(min_y, max_y, dx)
z = np.arange(min_z, max_z, dx)

grid = [x, y, z]

meshgrid = np.meshgrid(x, y, z)
ground_truth = np.zeros((len(x), len(y), len(z)))

test_dipole = np.array([0, 0 ,0])
test_dipole_dir = np.array([1, 0, 0])
seperation = 0.001
size = 0.005

posp = test_dipole + test_dipole_dir * seperation
posn = test_dipole - test_dipole_dir * seperation

pos_s = GaussianSourceKCSD3D(posp[0], posp[1], posp[2], size, conductivity=0.33)
neg_s = GaussianSourceKCSD3D(posn[0], posn[1], posn[2], size, conductivity=0.33)

ground_truth += pos_s.csd(*meshgrid)
ground_truth -= neg_s.csd(*meshgrid)

solution = solver.solve(grid, ground_truth)

import IPython
IPython.embed()

potential = solver.sample_solution(sampling_points)

meshgird_collate = np.rollaxis(np.array(meshgrid), 0, 4)
potential_grid = solver.sample_solution(meshgird_collate)


