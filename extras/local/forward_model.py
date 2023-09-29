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
import numpy as np
import dolfin

from .fem import common as fc


class _Base(object):
    # XXX: duplicated code with FEM classes
    def __init__(self, mesh, degree, config,
                 quiet=True,
                 ground_potential=None,
                 element_type="CG"):
        self.quiet=quiet
        self.set_ground_potential(ground_potential)
        self.fm = fc.FunctionManager(mesh, degree, element_type)
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
                                             dolfin.Constant(self.ground_potential),
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
        if not self.quiet:
            solver.parameters["monitor_convergence"] = True

        solver.solve(A, potential.vector(), b)

        return potential

    @property
    def CONDUCTIVITY(self):
        for section in self.config.sections():
            if self._is_conductive_volume(section):
                yield (self.config.getint(section, "volume"),
                       self.config.getfloat(section, "conductivity"))


class Spherical(_Base):
    def __init__(self, mesh, degree, config,
                 quiet=True,
                 ground_potential=0.0,
                 element_type="CG",
                 grounded_plate_at=-0.088):
        super().__init__(mesh, degree, config,
                         quiet=quiet,
                         ground_potential=ground_potential,
                         element_type=element_type)
        self.grounded_plate_at = grounded_plate_at

    @property
    def _grounded_boundary_tester(self):
        return lambda x, on_boundary: on_boundary and x[2] <= self.grounded_plate_at

    def set_ground_potential(self, value):
        self.ground_potential = value


class Slice(_Base):
    @property
    def _grounded_boundary_tester(self):
        return lambda x, on_boundary: on_boundary and x[2] > 0

    def set_ground_potential(self, value):
        self.ground_potential = (self._potential_at_dome()
                                 if value is None
                                 else value)

    def _potential_at_dome(self):
        if self.ground_potential is not None:
            return self.ground_potential

        radius = self.config.getfloat("dome", "radius")
        saline_conductivity = self.config.getfloat("saline", "conductivity")
        return 0.5 / (np.pi * saline_conductivity * radius)
