import os.path
import numpy as np
import pandas as pd

from kesi.common import GaussianSourceKCSD3D, FourSphereModel
from kesi.mfem_solver.forward_solver import CSDForwardSolver

this_folder = os.path.dirname(__file__)


meshfile_vaccuum = os.path.join(this_folder, "data", "generated", "meshes", "four_spheres_with_electrode_0.005.msh")
meshfile_air = os.path.join(this_folder, "data", "generated", "meshes", "four_spheres_in_air_with_plane_0.005.msh")
conductivities = np.array([0.33, 1.65, 0.0165, 0.33, 1e-10])

# conductivities = np.array([0.33, 1.65, 0.0165, 0.33, 1e-99])

electrodes_file = os.path.join(this_folder, "data", "generated", "tutorial", "four_spheres", "tutorial_electrodes_four.csv")

N = 1000
sampling_points_x = np.array([0,] * N)
sampling_points_y = np.array([0,] * N)
sampling_points_z = np.linspace(-0.1 , 0.1 ,N)
sampling_points = pd.read_csv(electrodes_file)[["X", "Y", "Z"]].values
sampling_points = np.stack([sampling_points_x, sampling_points_y, sampling_points_z]).T

solver_vaccuum = CSDForwardSolver(meshfile_vaccuum, conductivities, additional_refinement=True)
solver_air = CSDForwardSolver(meshfile_air, conductivities, additional_refinement=True)

mesh_points = np.array(solver_air.mesh.GetVertexArray())

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

test_dipole = np.array([0, 0 , 0.07])
test_dipole_dir = np.array([0, 0, 1])
seperation = 0.001
size = 0.001

posp = test_dipole + test_dipole_dir * seperation
posn = test_dipole - test_dipole_dir * seperation

pos_s = GaussianSourceKCSD3D(posp[0], posp[1], posp[2], size, conductivity=0.33)
neg_s = GaussianSourceKCSD3D(posn[0], posn[1], posn[2], size, conductivity=0.33)

ground_truth += pos_s.csd(*meshgrid)
ground_truth -= neg_s.csd(*meshgrid)

solver_air.solve(grid, ground_truth)
solver_vaccuum.solve(grid, ground_truth)



potential_mfem_air = np.squeeze(solver_air.sample_solution(sampling_points))
potential_mfem_vaccuum = np.squeeze(solver_vaccuum.sample_solution(sampling_points))

conductivities_analytical = FourSphereModel.Properties(*[0.33, 1.65, 0.0165, 0.33])
radii_analytical = FourSphereModel.Properties(*[0.079, 0.082, 0.086, 0.090])
dipole_moment_x = np.sum((meshgrid[0]-test_dipole[0]) * ground_truth * dx)
dipole_moment_y = np.sum((meshgrid[1]-test_dipole[1]) * ground_truth * dx)
dipole_moment_z = np.sum((meshgrid[2]-test_dipole[2]) * ground_truth * dx)
dipole_moment = np.array([dipole_moment_x, dipole_moment_y, dipole_moment_z]) * 1e-10

analytical_model_vacuum = FourSphereModel(conductivities_analytical, radii_analytical, n=1000)
dipole_analytical = analytical_model_vacuum(test_dipole, dipole_moment)

potential_dipole_analytical = np.squeeze(dipole_analytical(sampling_points[:, 0],
                                                           sampling_points[:, 1],
                                                           sampling_points[:, 2])
                                         )

import pylab as pb

pb.plot(sampling_points_z, potential_mfem_air)
pb.plot(sampling_points_z, potential_mfem_vaccuum)
pb.plot(sampling_points_z, potential_dipole_analytical)
pb.axvline(0.079, color='k', alpha=0.3)
pb.axvline(0.082, color='k', alpha=0.3)
pb.axvline(0.086, color='k', alpha=0.3)
pb.axvline(0.090, color='k', alpha=0.3)
pb.axvline(-0.079, color='k', alpha=0.3)
pb.axvline(-0.082, color='k', alpha=0.3)
pb.axvline(-0.086, color='k', alpha=0.3)
pb.axvline(-0.090, color='k', alpha=0.3)
pb.show()

import IPython
IPython.embed()



