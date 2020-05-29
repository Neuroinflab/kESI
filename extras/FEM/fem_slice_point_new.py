#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
#    Copyright (C) 2019-2020 Jakub M. Dzik (Laboratory of Neuroinformatics;   #
#    Nencki Institute of Experimental Biology of Polish Academy of Sciences)  #
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
import logging
import os

import numpy as np

try:
    from . import _fem_common as fc
    # When run as script raises:
    #  - `ModuleNotFoundError(ImportError)` (Python 3.6-7), or
    #  - `SystemError` (Python 3.3-5), or
    #  - `ValueError` (Python 2.7).

except (ImportError, SystemError, ValueError):
    import _fem_common as fc


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


try:
    from dolfin import (Constant, Mesh, MeshFunction, FunctionSpace,
                        TestFunction, TrialFunction, Function,
                        Measure, inner, grad, assemble, KrylovSolver,
                        Expression, DirichletBC, XDMFFile, MeshValueCollection,
                        cpp, HDF5File, MPI)

except (ModuleNotFoundError, ImportError):
    logger.warning("Unable to import from dolfin")

else:
    class SlicePointSourcePotentialFEM(object):
        MAX_ITER = 1000

        def __init__(self, mesh, config):
            self._load_mesh(mesh)
            self._load_config(config)
            self._degree = None
            self.global_preprocessing_time = fc.Stopwatch()
            self.local_preprocessing_time = fc.Stopwatch()
            self.solving_time = fc.Stopwatch()

        def _load_mesh(self, mesh):
            with XDMFFile(mesh + '.xdmf') as fh:
                self._mesh = Mesh()
                fh.read(self._mesh)

            self._boundaries = self._load_mesh_data(mesh + '_boundaries.xdmf',
                                                    "boundaries",
                                                    2)
            self._subdomains = self._load_mesh_data(mesh + '_subdomains.xdmf',
                                                    "subdomains",
                                                    3)

        def _load_mesh_data(self, path, name, dim):
            with XDMFFile(path) as fh:
                mvc = MeshValueCollection("size_t", self._mesh, dim)
                fh.read(mvc, name)
                return cpp.mesh.MeshFunctionSizet(self._mesh, mvc)

        def _load_config(self, config):
            self.config = configparser.ConfigParser()
            self.config.read(config)

        def _set_degree(self, degree):
            if degree != self._degree:
                self._degree = degree
                with self.global_preprocessing_time:
                    logger.debug('Creating function space...')
                    self._V = FunctionSpace(self._mesh, "CG", degree)
                    logger.debug('Done.  Creating integration subdomains...')
                    self.create_integration_subdomains()
                    logger.debug('Done.  Creating test function...')
                    self._v = TestFunction(self._V)
                    logger.debug('Done.  Creating potential function...')
                    self._potential_function = Function(self._V)
                    logger.debug('Done.  Creating trial function...')
                    self._potential_trial = TrialFunction(self._V)
                    logger.debug('Done.  Creating LHS part of equation...')
                    self._a = self._lhs()
                    logger.debug('Done.  Creating base potential formula...')
                    self._base_potential_expression = self._potential_expression()
                    logger.debug('Done.  Creating solver...')
                    self._solver = KrylovSolver("cg", "ilu")
                    self._solver.parameters["maximum_iterations"] = self.MAX_ITER
                    self._solver.parameters["absolute_tolerance"] = 1E-8
                    logger.debug('Done.  Solver created.')


        def _potential_expression(self, conductivity=0.0):
            return Expression('''
                              0.25
                              / ({pi}
                                 * conductivity
                                 * sqrt((x[0] - src_x)*(x[0] - src_x)
                                        + (x[1] - src_y)*(x[1] - src_y)
                                        + (x[2] - src_z)*(x[2] - src_z)))
                              '''.format(pi=np.pi),
                              degree=self._degree,
                              domain=self._mesh,
                              src_x=0.0,
                              src_y=0.0,
                              src_z=0.0,
                              conductivity=conductivity)

        def create_integration_subdomains(self):
            self._dx = Measure("dx")(subdomain_data=self._subdomains)
            self._ds = Measure("ds")(subdomain_data=self._boundaries)

        def _lhs(self):
            return sum(inner(Constant(c) * grad(self._potential_trial),
                             grad(self._v)) * self._dx(x)
                       for x, c in self.CONDUCTIVITY)

        @property
        def CONDUCTIVITY(self):
            for section in self.config.sections():
                if self._is_conductive_volume(section):
                    yield (self.config.getint(section, 'volume'),
                           self.config.getfloat(section, 'conductivity'))

        def _is_conductive_volume(self, section):
            return self.config.has_option(section, 'volume') and self.config.has_option(section, 'conductivity')

        def __call__(self, x, y, z):
            with self.local_preprocessing_time:
                logger.debug('Creating RHS...')
                L = self._rhs(x, y, z)
                logger.debug('Done.  Assembling linear equation vector...')
                known_terms = assemble(L)
                logger.debug('Done.  Assembling linear equation matrix...')
                self._terms_with_unknown = assemble(self._a)
                logger.debug('Done.  Defining boundary condition...')
                self._dirichlet_bc = self._boundary_condition(x, y, z)
                logger.debug('Done.  Applying boundary condition to the matrix...')
                self._dirichlet_bc.apply(self._terms_with_unknown)
                logger.debug('Done.  Applying boundary condition to the vector...')
                self._dirichlet_bc.apply(known_terms)
                logger.debug('Done.')

            try:
                logger.debug('Solving linear equation...')
                with self.solving_time:
                    self._solve(known_terms)

                logger.debug('Done.')
                return self._potential_function

            except RuntimeError as e:
                self.iterations = self.MAX_ITER
                logger.warning("Solver failed: {}".format(repr(e)))
                return None

        def _rhs(self, x, y, z):
            base_conductivity = self.base_conductivity(x, y, z)
            self._base_potential_expression.conductivity = base_conductivity
            self._base_potential_expression.src_x = x
            self._base_potential_expression.src_y = y
            self._base_potential_expression.src_z = z
            return -sum((inner((Constant(c - base_conductivity)
                                * grad(self._base_potential_expression)),
                               grad(self._v))
                         * self._dx(x)
                         for x, c in self.CONDUCTIVITY
                         if c != base_conductivity))
                        # # Eq. 18 at Piastra et al 2018

        def base_conductivity(self, x, y, z):
            return self.config.getfloat('slice', 'conductivity')

        def _boundary_condition(self, x, y, z):
            approximate_potential = self._potential_expression(self.config.getfloat('saline',
                                                                                    'conductivity'))
            radius = self.config.getfloat('dome', 'radius')
            return DirichletBC(self._V,
                               Constant(approximate_potential(0, 0, radius)
                                        - self._base_potential_expression(0, 0, radius)),
                               self._boundaries,
                               self.config.getint('dome', 'surface'))

        def _solve(self, known_terms):
            self.iterations = self._solver.solve(
                                              self._terms_with_unknown,
                                              self._potential_function.vector(),
                                              known_terms)


class FunctionManager(object):
    """
    TODO: Rewrite to:
    - use INI configuration with mesh name and degree,
    - use lazy mesh loading.
    """
    def __init__(self, mesh, degree):
        with XDMFFile(mesh + '.xdmf') as fh:
            self._mesh = Mesh()
            fh.read(self._mesh)

        self._V = FunctionSpace(self._mesh, "CG", degree)

    def write(self, filename, function, name):
        with HDF5File(MPI.comm_self,
                      filename,
                      'a' if os.path.exists(filename) else 'w') as fh:
            fh.write(function, name)

    def read(self, filename, name):
        function = Function(self._V)
        with HDF5File(MPI.comm_self, filename, 'a') as fh:
            fh.read(function, name)

        return function


# TODO:
# Create Romberg Function manager/controler and Romberg function factory.