#!/usr/bin/env python
# coding: utf-8

import argparse
import csv

import numpy as np
import scipy.integrate as si

from dolfin import (Expression, Measure, inner, grad, assemble,
                    Constant, KrylovSolver, DirichletBC)
import dolfin

import FEM.fem_sphere_point_new as fspn
import FEM.fem_common as fc

import common as common


class NegativePotential(dolfin.UserExpression):
    def __init__(self, potential, *args, **kwargs):
        self._potential = potential
        super(NegativePotential, self).__init__(*args, **kwargs)

    def eval(self, value, x):
        value[0] = self._potential(*x)


class SphericalModelFEM(object):
    def __init__(self, fem, grounded_plate_at):
        self.GROUNDED_PLATE_AT = grounded_plate_at
        self.fem = fem
        self.CONDUCTIVITY = list(fem.CONDUCTIVITY)
        self.BOUNDARY_CONDUCTIVITY = list(fem.BOUNDARY_CONDUCTIVITY)

        self.solver = KrylovSolver("cg", "ilu")
        self.solver.parameters["maximum_iterations"] = 1000
        self.solver.parameters["absolute_tolerance"] = 1E-8

        self.V = fem._fm.function_space
        self.v = fem._fm.test_function()
        self.u = fem._fm.trial_function()
        self.dx = fem._dx
        self.ds = fem._ds

    def function(self):
        return self.fem._fm.function()

    def reciprocal_correction_potential(self, x, y, z):
        """
        .. deprecated::
                   Moved to `SphereOnGroundedPlatePointSourcePotentialFEM` class
                   (rewritten) and available as its `.solve()` method.
                   Preserved as the original code not obfuscated by the
                   `_SubtractionPointSourcePotentialFEM` class protocol.

        Parameters
        ----------
        x
        y
        z

        Returns
        -------

        """
        dx_src = f'(x[0] - src_x)'
        dy_src = f'(x[1] - src_y)'
        dz_src = f'(x[2] - src_z)'

        r_src2 = f'({dx_src} * {dx_src} + {dy_src} * {dy_src} + {dz_src} * {dz_src})'
        r_src = f'sqrt({r_src2})'
        r_sphere2 = '(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])'

        dot_src = f'({dx_src} * x[0] + {dy_src} * x[1] + {dz_src} * x[2]) / sqrt({r_src2} * {r_sphere2})'
        potential_exp = Expression(f'''
                                    {0.25 / np.pi} / conductivity
                                    * (1.0 / {r_src})
                                    ''',
                                   degree=self.fem.degree,
                                   domain=self.fem._fm._mesh,
                                   conductivity=0.33,
                                   src_x=0.0,
                                   src_y=0.0,
                                   src_z=0.0)
        minus_potential_exp = Expression(f'''
                                        {-0.25 / np.pi} / conductivity
                                        * (1.0 / {r_src})
                                        ''',
                                         degree=self.fem.degree,
                                         domain=self.fem._fm._mesh,
                                         conductivity=0.33,
                                         src_x=0.0,
                                         src_y=0.0,
                                         src_z=0.0)
        potential_grad_dot = Expression(f'''
                                        x[2] >= {self.GROUNDED_PLATE_AT} ?
                                        -1 * {-0.25 / np.pi} / conductivity
                                        * ({dot_src} / {r_src2})
                                        : 0
                                        ''',
                                        degree=self.fem.degree,
                                        domain=self.fem._fm._mesh,
                                        conductivity=0.33,
                                        src_x=0.0,
                                        src_y=0.0,
                                        src_z=0.0)

        conductivity = fem.base_conductivity(x, y, z)

        for expr in [potential_exp, potential_grad_dot, minus_potential_exp]:
            expr.conductivity = conductivity
            expr.src_x = x
            expr.src_y = y
            expr.src_z = z

        print(' solving')

        a = sum(inner(Constant(c) * grad(self.u), grad(self.v)) * self.dx(i)
                for i, c in self.CONDUCTIVITY)
        L = (-sum(inner((Constant(c - conductivity)
                         * grad(potential_exp)),
                        grad(self.v))
                  * self.dx(i)
                  for i, c in self.CONDUCTIVITY
                  if c != conductivity)
             + sum(Constant(c)
                   * potential_grad_dot * self.v * self.ds(i)
                   for i, c in self.BOUNDARY_CONDUCTIVITY)
             )

        return self.solve(L, a, minus_potential_exp)

    def source_potential(self, csd=None, src=None):
        if csd is None:
            csd = self.callable_as_function(src.csd)

        a = sum(Constant(c)
                * inner(grad(self.u),
                        grad(self.v))
                * self.dx(i)
                for i, c in self.CONDUCTIVITY)
        L = csd * self.v * self.dx

        return self.solve(L, a, Constant(0))

    def source_correction(self, src):
        potential_f = self.callable_as_function(src.potential)

        conductivity = self.fem.base_conductivity(x_ele, y_ele, z_ele)

        print(' solving')

        a = sum(inner(Constant(c) * grad(self.u), grad(self.v)) * self.dx(i)
                for i, c in self.CONDUCTIVITY)
        L = (-sum(inner((Constant(c - conductivity)
                         * grad(potential_f)),
                        grad(self.v))
                  * self.dx(i)
                  for i, c in self.CONDUCTIVITY
                  if c != conductivity)
             - sum(Constant(c)
                   * inner(grad(potential_f),
                           dolfin.FacetNormal(self.fem._fm.mesh))
                   * self.v * self.ds(i)
                   for i, c in self.BOUNDARY_CONDUCTIVITY)
             )
        neg_potential_f = NegativePotential(src.potential, self.fem._fm.mesh)
        return self.solve(L, a, neg_potential_f)

    def callable_as_function(self, f):
        n = self.V.dim()
        d = self.fem._fm.mesh.geometry().dim()
        dof_coordinates = self.V.tabulate_dof_coordinates()
        dof_coordinates.resize((n, d))
        dof_x = dof_coordinates[:, 0]
        dof_y = dof_coordinates[:, 1]
        dof_z = dof_coordinates[:, 2]
        g = fem._fm.function()
        g.vector()[:] = f(dof_x, dof_y, dof_z)
        return g

    def solve(self, L, a, plate_potential):
        A = assemble(a)
        b = assemble(L)
        # print(' assembled')

        dirichlet_bc = DirichletBC(self.V,
                                   plate_potential,
                                   (lambda x, on_boundary:
                                    on_boundary
                                    and x[2] < self.GROUNDED_PLATE_AT))

        dirichlet_bc.apply(A, b)
        # print(' modified')
        f = self.function()
        self.solver.solve(A, f.vector(), b)
        # print(' solved')
        return f


class LeadfieldIntegrator(object):
    def __init__(self, model):
        self.model = model

    def fenics(self, leadfield,
               csd=None,
               src=None):
        if csd is None:
            csd = self.model.callable_as_function(src.csd)

        return assemble(csd * leadfield * self.model.dx)

    def romberg(self, leadfield, src, k=4):
        r = max(src._nodes)
        n = 2 ** k + 1
        dxyz = r ** 3 / 2 ** (3 * k - 3)
        X = np.linspace(src.x - r,
                        src.x + r,
                        n)
        Y = np.linspace(src.y - r,
                        src.y + r,
                        n)
        Z = np.linspace(src.z - r,
                        src.z + r,
                        n)
        CSD = src.csd(X.reshape(-1, 1, 1),
                      Y.reshape(1, -1, 1),
                      Z.reshape(1, 1, -1))

        return dxyz * si.romb([si.romb([si.romb([leadfield(xx, yy, zz)
                                                  for zz in Z] * CSD_XY)
                                        for yy, CSD_XY in zip(Y, CSD_X)])
                               for xx, CSD_X in zip(X, CSD)])

    def _legacy_romberg_by_convotulion(self, leadfield, src, k=4):
        from kesi.kernel.constructor import Convolver
        r = max(src._nodes)
        n = 2 ** k + 1

        X = np.linspace(src.x - r,
                        src.x + r,
                        n)
        Y = np.linspace(src.y - r,
                        src.y + r,
                        n)
        Z = np.linspace(src.z - r,
                        src.z + r,
                        n)

        convolver = Convolver([X, Y, Z],
                              [X, Y, Z])

        model_src = common.SphericalSplineSourceKCSD(0, 0, 0,
                                                     src._nodes,
                                                     src._csd_polynomials,
                                                     src.conductivity)

        LEADFIELD = np.full([n, n, n], np.nan)

        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                for k, z in enumerate(Z):
                    LEADFIELD[i, j, k] = leadfield(x, y, z)

        weights = si.romb(np.identity(n), dx=r / (n // 2))
        POT_CORR = convolver.leadfield_to_potential_basis_functions(LEADFIELD,
                                                                    model_src.csd,
                                                                    (weights,) * 3)
        return POT_CORR[n // 2, n // 2, n // 2]


parser = argparse.ArgumentParser(description='Test different methods of potential calculation.')
parser.add_argument('configs',
                    metavar='<config.ini>',
                    # dest='configs',
                    nargs='+',
                    help='FEM configs')
parser.add_argument('-o', '--output',
                    metavar='<output.csv>',
                    dest='output',
                    help='path to the output file',
                    default=None)
parser.add_argument('-q', '--quiet',
                    dest='quiet',
                    action='store_true',
                    help='do not print results',
                    default=False)
parser.add_argument('-k', '--k-romberg',
                    type=int,
                    dest='k',
                    metavar="<Romberg's method k>",
                    help="k parameter of the Romberg's method",
                    default=4)
parser.add_argument('-g', '--grounded_plate_edge_z',
                    type=float,
                    dest='grounded_plate_edge_z',
                    metavar="<grounded plate edge's z>",
                    help='Z coordinate of the grounded plate',
                    default=-0.088)
parser.add_argument('-e', '--electrode-location',
                    type=float,
                    nargs=3,
                    dest='electrode',
                    metavar=("<electrode's X>",
                             "<electrode's Y>",
                             "<electrode's Z>",
                             ),
                    help='XYZ coordinates of the electrode',
                    default=[-0.0532907,
                             -0.0486064,
                             0.016553])
parser.add_argument('-s', '--source-location',
                    type=float,
                    nargs=3,
                    dest='source',
                    metavar=("<source's X>",
                             "<source's Y>",
                             "<source's Z>",
                             ),
                    help='XYZ coordinates of the source centroid',
                    default=[-0.0532907 + 4 * 8e-4,
                             -0.0486064 + 4 * 8e-4,
                              0.016553 - 4 * 8e-4])
parser.add_argument('-r', '--source_radius',
                    type=float,
                    dest='source_radius',
                    metavar="<source's r>",
                    help='radius of the source',
                    default=3 * 8e-4)

args = parser.parse_args()

STANDARD_DEVIATION = args.source_radius / 3
SPLINE_NODES = [STANDARD_DEVIATION, 3 * STANDARD_DEVIATION]
SPLINE_COEFFS = [[1],
                 [0,
                  2.25 / STANDARD_DEVIATION,
                  -1.5 / STANDARD_DEVIATION ** 2,
                  0.25 / STANDARD_DEVIATION ** 3]]

x_ele, y_ele, z_ele = args.electrode
x_src, y_src, z_src = args.source


HEADER = ['FEM',
          'approximate',
          'correction',
          'reciprocal_correction',
          'reciprocal_correction_romberg',
          'config',
          ]

writer = None
if args.output is not None:
    writer = csv.writer(open(args.output, 'w', newline=''))
    writer.writerow(HEADER)

if not args.quiet:
    print(f'Electrode at: {x_ele} {y_ele} {z_ele}')
    print(f'Source at: {x_src} {y_src} {z_src}')
    print()
    print('    FEM\t'
          '  appr. (err %)\t'
          ' subtr. (err %)\t'
          '     rec. sub. (err. %)\t'
          'rec. sub. int. (err. %)\t'
          'config')


for config in args.configs:
    configuration = fc.LegacyConfigParser(config)
    fem = fspn.SphereOnGroundedPlatePointSourcePotentialFEM(
                    fc.FunctionManager(configuration.getpath('fem', 'mesh'),
                                       configuration.getint('fem', 'degree'),
                                       configuration.get('fem', 'element_type')),
                    configuration.getpath('fem', 'config'),
                    grounded_plate_edge_z=args.grounded_plate_edge_z)
    model = SphericalModelFEM(fem, grounded_plate_at=args.grounded_plate_edge_z)
    integrator = LeadfieldIntegrator(model)

    conductivity = fem.base_conductivity(x_src, y_src, z_src)
    src = common.SphericalSplineSourceKCSD(x_src, y_src, z_src,
                                           SPLINE_NODES,
                                           SPLINE_COEFFS,
                                           conductivity)

    v_appr = src.potential(x_ele, y_ele, z_ele)

    csd_f = model.callable_as_function(src.csd)
    potential_fem = model.source_potential(csd=csd_f)
    v_fem = potential_fem(x_ele, y_ele, z_ele)

    correction = model.source_correction(src)
    v_corr = correction(x_ele, y_ele, z_ele)

    leadfield_corr_ele = fem.solve(x_ele, y_ele, z_ele)
    v_corr_rec = integrator.fenics(leadfield_corr_ele,
                                   csd=csd_f)

    v_corr_rec_int = integrator.romberg(leadfield_corr_ele, src, args.k)

    if writer is not None:
        writer.writerow(map(str,
                            [v_fem,
                             v_appr,
                             v_corr,
                             v_corr_rec,
                             v_corr_rec_int,
                             config]))

    if not args.quiet:
        v_sub = v_appr + v_corr
        v_sub_rec = v_appr + v_corr_rec
        v_sub_rec_int = v_appr + v_corr_rec_int
        print(f'{round(v_fem):>7d}\t'
              f'{round(v_appr):>7d} '
              f'({100 * (v_appr / v_fem - 1):.2g})\t'
              f'{round(v_sub):>7d} '
              f'({100 * (v_sub / v_fem - 1):.2g})\t'
              f'        {round(v_sub_rec):>7d} '
              f'({100 * (v_sub_rec / v_fem - 1):.2g})\t'
              f'        {round(v_sub_rec_int):>7d} '
              f'({100 * (v_sub_rec_int / v_fem - 1):.2g})\t'
              f'{config}'
              )

