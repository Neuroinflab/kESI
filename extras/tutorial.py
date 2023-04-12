#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
#    Copyright (C) 2021-2023 Jakub M. Dzik (Laboratory of Neuroinformatics;   #
#    Nencki Institute of Experimental Biology of Polish Academy of Sciences)  #
#    Copyright (C) 2021-2023 Jakub M. Dzik (Institute of Applied Psychology;  #
#    Faculty of Management and Social Communication; Jagiellonian University) #
#                                                                             #
#    This software is free software: you can redistribute it and/or modify    #
#    it under the terms of the GNU General Public License as published by     #
#    the Free Software Foundation, either version 3 of the License, or        #
#    (at your option) any later version.                                      #
#                                                                             #
#    This software is distributed in the hope that it will be useful,         #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
#    GNU General Public License for more details.                             #
#                                                                             #
#    You should have received a copy of the GNU General Public License        #
#    along with this software.  If not, see http://www.gnu.org/licenses/.     #
#                                                                             #
###############################################################################

import configparser
import dolfin

import FEM.fem_common as fc


class _ForwardModelBase(object):
    # XXX: duplicated code with FEM classes

    def __init__(self, mesh, degree, config):
        self.fm = fc.FunctionManager(mesh, degree, "CG")
        self.config = configparser.ConfigParser()
        self.config.read(config)

        self.V = self.fm.function_space
        mesh = self.fm.mesh

        n = self.V.dim()
        d = mesh.geometry().dim()

        self.dof_coords = self.V.tabulate_dof_coordinates()
        self.dof_coords.resize((n, d))

        self.csd_f = self.fm.function()

        self.subdomains = self.fm.load_subdomains()
        self.dx = dolfin.Measure("dx")(subdomain_data=self.subdomains)

    def _is_conductive_volume(self, section):
        return (self.config.has_option(section, "volume")
                and self.config.has_option(section, "conductivity"))

    def __call__(self, csd_interpolator):
        self.csd_f.vector()[:] = csd_interpolator(self.dof_coords)

        dirichlet_bc_gt = dolfin.DirichletBC(self.V,
                                             dolfin.Constant(0),
                                             self._grounded_boundary_tester)
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
        solver.solve(A, potential.vector(), b)

        return potential

    @property
    def CONDUCTIVITY(self):
        for section in self.config.sections():
            if self._is_conductive_volume(section):
                yield (self.config.getint(section, "volume"),
                       self.config.getfloat(section, "conductivity"))


class SphericalForwardModel(_ForwardModelBase):
    GROUNDED_PLATE_AT = -0.088

    @property
    def _grounded_boundary_tester(self):
        return lambda x, on_boundary: on_boundary and x[2] <= self.GROUNDED_PLATE_AT


class SliceForwardModel(_ForwardModelBase):
    @property
    def _grounded_boundary_tester(self):
        return lambda x, on_boundary: on_boundary and x[2] > 0
