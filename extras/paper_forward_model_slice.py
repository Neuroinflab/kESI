#!/usr/bin/env python
# coding: utf-8

import argparse
import configparser

import numpy as np
import pandas as pd
import scipy.interpolate as si
import dolfin

import FEM.fem_common as fc


class ForwardModel(object):
    # XXX: duplicated code with FEM classes
    def __init__(self, mesh, degree, config,
                 quiet=True,
                 ground_potential=None,
                 element_type='CG'):
        self.quiet=quiet
        self.ground_potential = ground_potential
        self.fm = fc.FunctionManager(mesh, degree, element_type)
        self.config = configparser.ConfigParser()
        self.config.read(config)

        mesh_filename = mesh[:-5]

        self.V = self.fm.function_space
        mesh = self.fm.mesh

        n = self.V.dim()
        d = mesh.geometry().dim()

        self.dof_coords = self.V.tabulate_dof_coordinates()
        self.dof_coords.resize((n, d))

        self.csd_f = self.fm.function()

        with dolfin.XDMFFile(mesh_filename + '_subdomains.xdmf') as fh:
            mvc = dolfin.MeshValueCollection("size_t", mesh, 3)
            fh.read(mvc, "subdomains")
            self.subdomains = dolfin.cpp.mesh.MeshFunctionSizet(mesh, mvc)
            self.dx = dolfin.Measure("dx")(subdomain_data=self.subdomains)

    @property
    def CONDUCTIVITY(self):
        for section in self.config.sections():
            if self._is_conductive_volume(section):
                yield (self.config.getint(section, 'volume'),
                       self.config.getfloat(section, 'conductivity'))

    def _is_conductive_volume(self, section):
        return (self.config.has_option(section, 'volume')
                and self.config.has_option(section, 'conductivity'))

    def __call__(self, csd_interpolator):
        self.csd_f.vector()[:] = csd_interpolator(self.dof_coords)

        dirichlet_bc_gt = dolfin.DirichletBC(self.V,
                                             dolfin.Constant(self._potential_at_dome()),
                                             (lambda x, on_boundary:
                                              on_boundary and x[2] > 0))
        test = self.fm.test_function()
        trial = self.fm.trial_function()
        potential = self.fm.function()

        dx = self.dx
        a = sum(dolfin.Constant(c)
                * dolfin.inner(dolfin.grad(trial),
                               dolfin.grad(test))
                * dx(i)
                for i, c
                in self.CONDUCTIVITY)
        L = self.csd_f * test * dx

        b = dolfin.assemble(L)
        A = dolfin.assemble(a)
        dirichlet_bc_gt.apply(A, b)

        solver = dolfin.KrylovSolver("cg", "ilu")
        solver.parameters["maximum_iterations"] = 10000
        solver.parameters["absolute_tolerance"] = 1E-8
        if not self.quiet:
            solver.parameters["monitor_convergence"] = True

        solver.solve(A, potential.vector(), b)

        return potential

    def _potential_at_dome(self):
        if self.ground_potential is not None:
            return self.ground_potential

        radius = self.config.getfloat('dome', 'radius')
        saline_conductivity = self.config.getfloat('saline', 'conductivity')
        return 0.5 / (np.pi * saline_conductivity * radius)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model CSD in sphere on plate geometry with FEM.')
    parser.add_argument('-o', '--output',
                        metavar='<images.csv>',
                        dest='output',
                        required=True,
                        help='path to the images file')
    parser.add_argument('-s', '--sources',
                        metavar='<sources.npz>',
                        dest='sources',
                        required=True,
                        help='path to the file containing sources (CSD)')
    parser.add_argument('-e', '--electrodes',
                        metavar='<electrodes.csv>',
                        dest='electrodes',
                        required=True,
                        help='path to the electrode location config file')
    parser.add_argument('-c', '--config',
                        metavar='<config.ini>',
                        dest='config',
                        required=True,
                        help='path to the model config file')
    parser.add_argument('-m', '--mesh',
                        metavar='<mesh.xdmf>',
                        dest='mesh',
                        required=True,
                        help='path to the FEM mesh')
    parser.add_argument('-d', '--degree',
                        type=int,
                        metavar='<FEM element degree>',
                        dest='degree',
                        help='degree of FEM elements',
                        default=1)
    parser.add_argument('--element-type',
                        metavar='<FEM element type>',
                        dest='element_type',
                        help='type of FEM elements',
                        default='CG')
    parser.add_argument('-g', '--ground-potential',
                        type=float,
                        dest='ground_potential',
                        metavar="<ground potential>",
                        help='the potential at the grounded slice-covering dome')
    parser.add_argument('-q', '--quiet',
                        dest='quiet',
                        action='store_true',
                        help='do not print results',
                        default=False)
    parser.add_argument('--start-from',
                        type=int,
                        metavar='<source number>',
                        dest='start_from',
                        help='number of the first source to start from (useful in case of broken run)',
                        default=0)

    args = parser.parse_args()


    DF = pd.read_csv(args.electrodes)
    ELECTRODE_LOCATION = list(zip(DF.X, DF.Y, DF.Z))

    fem = ForwardModel(args.mesh, args.degree, args.config,
                       element_type=args.element_type,
                       ground_potential=args.ground_potential,
                       quiet=args.quiet)

    with np.load(args.sources) as fh:
        XYZ = [fh[x].flatten() for x in 'XYZ']
        CSD = fh['CSD']
        for i in range(args.start_from, CSD.shape[-1]):
            csd = si.RegularGridInterpolator(XYZ, CSD[:, :, :, i],
                                             bounds_error=False,
                                             fill_value=0)
            potential = fem(csd)
            DF[f'SOURCE_{i}'] = [potential(*xyz) for xyz in ELECTRODE_LOCATION]
            DF.to_csv(args.output,
                      index=False)
