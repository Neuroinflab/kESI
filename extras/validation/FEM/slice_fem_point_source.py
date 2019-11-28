"""
@author: mbejtka
"""
import os
import numpy as np
import gc
from dolfin import (Constant, Mesh, MeshFunction, FunctionSpace, TestFunction,
                    TrialFunction, Function, Measure, inner, grad, assemble,
                    KrylovSolver, Point, PointSource, info)
from dolfin import Expression, DirichletBC
import parameters_fem as params


def extract_pots(phi, positions):
    compt_values = np.zeros(positions.shape[0])
    for ii in range(positions.shape[0]):
        compt_values[ii] = phi(positions[ii, :])
    return compt_values


def main_slice_fem(mesh, subdomains, boundaries, src_pos, snk_pos):
    sigma_ROI = Constant(params.sigma_roi)
    sigma_SLICE = Constant(params.sigma_slice)
    sigma_SALINE = Constant(params.sigma_saline)
    sigma_AIR = Constant(0.)

    V = FunctionSpace(mesh, "CG", 2)
    v = TestFunction(V)
    u = TrialFunction(V)

    phi = Function(V)
    dx = Measure("dx")(subdomain_data=subdomains)
    ds = Measure("ds")(subdomain_data=boundaries)
    a = inner(sigma_ROI * grad(u), grad(v))*dx(params.roivol) + \
        inner(sigma_SLICE * grad(u), grad(v))*dx(params.slicevol) + \
        inner(sigma_SALINE * grad(u), grad(v))*dx(params.salinevol)
    L = Constant(0)*v*dx
    A = assemble(a)
    b = assemble(L)

    x_pos, y_pos, z_pos = src_pos
    point = Point(x_pos, y_pos, z_pos)
    delta = PointSource(V, point, 1.)
    delta.apply(b)

    x_pos, y_pos, z_pos = snk_pos
    point1 = Point(x_pos, y_pos, z_pos)
    delta1 = PointSource(V, point1, -1.)
    delta1.apply(b)

    solver = KrylovSolver("cg", "ilu")
    solver.parameters["maximum_iterations"] = 1000
    solver.parameters["absolute_tolerance"] = 1E-8
    solver.parameters["monitor_convergence"] = True

    info(solver.parameters, True)
#    set_log_level(PROGRESS) does not work in fenics 2018.1.0
    solver.solve(A, phi.vector(), b)

    ele_pos_list = params.ele_coords
    vals = extract_pots(phi, ele_pos_list)
    # np.save(os.path.join('results', save_as), vals)
    return vals

if __name__ == '__main__':
    print('Loading meshes')
    mesh = Mesh(os.path.join("mesh", "slice_saline_model.xml"))
    subdomains = MeshFunction("size_t", mesh, os.path.join("mesh", "slice_saline_model_physical_region.xml"))
    boundaries = MeshFunction("size_t", mesh, os.path.join("mesh", "slice_saline_model_facet_region.xml"))
    for dipole in params.dipole_list:
        print('Now computing FEM for dipole: ', dipole['name'])
        src_pos = dipole['src_pos']
        snk_pos = dipole['snk_pos']
        print('Done loading meshes')
        fem_20 = main_slice_fem(mesh, subdomains, boundaries, src_pos, snk_pos)
        print('Done 4Shell-FEM-20')
        f = open(os.path.join('Numerical_' + dipole['name'] + '.npz'), 'wb')
        np.savez(f, fem_20=fem_20)
        f.close()