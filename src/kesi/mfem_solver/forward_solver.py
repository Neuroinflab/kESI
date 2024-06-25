import numpy as np
import pyvista
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

from kesi.mfem_solver.mfem_piecewise_solver import mfem_solve_mesh, csd_distribution_coefficient, prepare_mesh, \
    prepare_fespace
from mfem import ser as mfem

class CSDForwardSolver:
    def __init__(self, meshfile, conductivities, boundary_value=0, additional_refinement=False,
                 sampling_points=None):
        """
        meshfile - mfem compatable mesh file
        conductivities - numpy array of conductances per mesh material in S/m
        sampling points - numpy array (N, 3) of simulated electrodes positions, if provided now, will be used to refine mesh around those positions
        boundary_value - the value at boudaries, usually grounding electrode
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

    def solve(self, xyz, csd):
        """
        saves solution into self.solution as MFEM gridfunction and self.solution_interpolated for sampling at any point
        """
        coeff = csd_distribution_coefficient(xyz, csd)
        solution = mfem_solve_mesh(csd_coefficient=coeff,
                                   mesh=self.mesh,
                                   boundary_potential=self.boundary_value,
                                   conductivities=self.conductivities)
        self.solution = solution

        verts = np.array(self.mesh.GetVertexArray())
        sol = self.solution.GetDataArray()
        verts_triangulation = Delaunay(verts)
        self.solution_interpolated = LinearNDInterpolator(verts_triangulation, sol)
        return solution

    def sample_solution_probe(self, x, y, z):
        return self.solution_interpolated([x, y, z])[0]

    def sample_solution(self, positions):
        """positions - array (N, 3)"""
        assert self.solution_interpolated is not None
        pos_arr = np.array(positions)
        return self.solution_interpolated(pos_arr)
