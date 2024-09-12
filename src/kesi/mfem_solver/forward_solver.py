from functools import lru_cache

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

from kesi.mfem_solver.mfem_piecewise_solver import mfem_solve_mesh, csd_distribution_coefficient, prepare_mesh


@lru_cache
def _cachedDelaunay(verts):
    """verts as tuple of tuples"""
    verts = np.array(verts)
    verts_triangulation = Delaunay(verts)
    return verts_triangulation


def cachedDelaunay(verts):
    """
    Calculates delaunay triangulation of given vertices, with caching of the result in RAM

    verts - numpy array of verts (N, 3)
    """
    return _cachedDelaunay(tuple(map(tuple, verts)))


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

        verts = np.array(self.mesh.GetVertexArray())
        sol = self.solution.GetDataArray()
        verts_triangulation = cachedDelaunay(verts)

        self.solution_interpolated = self.interpolator(verts_triangulation, sol)
        return solution

    def solve(self, xyz, csd):
        """
        saves solution into self.solution as MFEM gridfunction and self.solution_interpolated for sampling at any point
        """
        coeff = csd_distribution_coefficient(xyz, csd)
        return self.solve_coeff(coeff)


    def sample_solution_probe(self, x, y, z):
        return self.solution_interpolated([x, y, z])[0]

    def sample_solution(self, positions):
        """positions - array (N, 3)"""
        assert self.solution_interpolated is not None
        pos_arr = np.array(positions)
        return self.solution_interpolated(pos_arr)
