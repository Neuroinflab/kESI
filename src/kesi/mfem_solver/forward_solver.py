from functools import lru_cache

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

from kesi.fem_utils.pyvista_resampling import convert_mfem_to_pyvista, pyvista_sample_points, pyvista_sample_grid
from kesi.mfem_solver.mfem_piecewise_solver import mfem_solve_mesh, csd_distribution_coefficient, prepare_mesh


class CSDForwardSolver:
    def __init__(self, meshfile, conductivities, boundary_value=0, additional_refinement=False,
                 sampling_points=None,
                 interpolator=LinearNDInterpolator):
        """
        meshfile - mfem compatable mesh file
        conductivities - numpy array of conductances per mesh material in S/m
        sampling points - numpy array (N, 3) of simulated electrodes positions, if provided now, will be used to refine mesh around those positions
        boundary_value - the value at boudaries, usually grounding electrode
        interpolator - can use from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        """
        self.meshfile = meshfile
        self.conductivities = conductivities
        self.boundary_value = boundary_value
        if sampling_points:
            self.mesh = prepare_mesh(self.meshfile, additional_refinement, sampling_points)
        else:
            self.mesh = prepare_mesh(self.meshfile, additional_refinement)
        self.solution = None
        self.solution_interpolated = None
        self.interpolator = interpolator

    def solve_coeff(self, coeff):
        solution = mfem_solve_mesh(csd_coefficient=coeff,
                                   mesh=self.mesh,
                                   boundary_potential=self.boundary_value,
                                   conductivities=self.conductivities)
        self.solution = solution

        self.pyvista_mesh_solution = convert_mfem_to_pyvista(self.mesh, [solution], ["pot"])
        return solution

    def solve(self, xyz, csd):
        """
        saves solution into self.solution as MFEM gridfunction and self.solution_interpolated for sampling at any point
        """
        coeff = csd_distribution_coefficient(xyz, csd)
        return self.solve_coeff(coeff)

    def sample_solution_probe(self, x, y, z):
        points = np.array([[x, y, z]])
        cloud = self.sample_solution(points)
        return cloud.get_array("pot")

    def sample_solution(self, positions):
        """positions - array (N, 3) or meshgrid stack [X, Y, Z, 3]"""

        cloud = pyvista_sample_points(self.pyvista_mesh_solution, positions)
        data = cloud.get_array("pot")
        return data

    def sample_grid(self, grid):
        sampled = pyvista_sample_grid(self.pyvista_mesh_solution, grid)
        data = sampled.get_array("pot")
        sampled_grid = data.reshape(grid.shape[0:3], order="F")
        return sampled_grid

