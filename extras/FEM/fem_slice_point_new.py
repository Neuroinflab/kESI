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
logger.setLevel(logging.INFO)


DIRNAME = os.path.dirname(__file__)


try:
    from dolfin import (Constant, Mesh, MeshFunction, FunctionSpace, FacetNormal,
                        TestFunction, TrialFunction, Function,
                        Measure, inner, grad, assemble, KrylovSolver,
                        Expression, DirichletBC, XDMFFile, MeshValueCollection,
                        cpp, HDF5File, MPI)

except (ModuleNotFoundError, ImportError):
    logger.warning("Unable to import from dolfin")

else:
    class SlicePointSourcePotentialFEM(object):
        MAX_ITER = 1000

        def __init__(self, config):
            self._fm = FunctionManagerINI(config)
            self._setup_mesh(self._fm.getpath('fem', 'mesh')[:-5])
            self._load_config(self._fm.getpath('fem', 'config'))
            self.global_preprocessing_time = fc.Stopwatch()
            self.local_preprocessing_time = fc.Stopwatch()
            self.solving_time = fc.Stopwatch()

            self._set_degree(self.degree)

        def _setup_mesh(self, mesh):
            self._boundaries = self._load_mesh_data(mesh + '_boundaries.xdmf',
                                                    "boundaries",
                                                    2)
            self._subdomains = self._load_mesh_data(mesh + '_subdomains.xdmf',
                                                    "subdomains",
                                                    3)
            self._facet_normal = FacetNormal(self._fm.mesh)

        def _load_mesh_data(self, path, name, dim):
            with XDMFFile(path) as fh:
                mvc = MeshValueCollection("size_t", self._fm.mesh, dim)
                fh.read(mvc, name)
                return cpp.mesh.MeshFunctionSizet(self._fm.mesh, mvc)

        def _load_config(self, config):
            self.config = configparser.ConfigParser()
            self.config.read(config)

        @property
        def degree(self):
            return self._fm.degree

        @degree.setter
        def degree(self, degree):
            if degree != self.degree:
                self._set_degree(degree)

        def _set_degree(self, degree):
            self._fm.degree = degree
            with self.global_preprocessing_time:
                logger.debug('Creating integration subdomains...')
                self.create_integration_subdomains()
                logger.debug('Done.  Creating test function...')
                self._v = self._fm.test_function()
                logger.debug('Done.  Creating potential function...')
                self._potential_function = self._fm.function()
                logger.debug('Done.  Creating trial function...')
                self._potential_trial = self._fm.trial_function()
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
                              degree=self.degree,
                              domain=self._fm.mesh,
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

        @property
        def BOUNDARY_CONDUCTIVITY(self):
            for section in self.config.sections():
                if self._is_conductive_boundary(section):
                    yield (self.config.getint(section, 'surface'),
                           self.config.getfloat(section, 'conductivity'))

        def _is_conductive_boundary(self, section):
            return self.config.has_option(section, 'surface') and self.config.has_option(section, 'conductivity')

        def solve(self, x, y, z):
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
            return (-sum((inner((Constant(c - base_conductivity)
                                 * grad(self._base_potential_expression)),
                                grad(self._v))
                          * self._dx(x)
                          for x, c in self.CONDUCTIVITY
                          if c != base_conductivity))
                          # # Eq. 18 at Piastra et al 2018
                    + sum(Constant(c)
                          * inner(self._facet_normal,
                                  grad(self._base_potential_expression))
                          * self._v
                          * self._ds(s)
                          for s, c in self.BOUNDARY_CONDUCTIVITY))

        def base_conductivity(self, x, y, z):
            return self.config.getfloat('slice', 'conductivity')

        def _boundary_condition(self, x, y, z):
            approximate_potential = self._potential_expression(self.config.getfloat('saline',
                                                                                    'conductivity'))
            radius = self.config.getfloat('dome', 'radius')
            return DirichletBC(self._fm.function_space,
                               Constant(approximate_potential(0, 0, radius)
                                        - self._base_potential_expression(0, 0, radius)),
                               self._boundaries,
                               self.config.getint('dome', 'surface'))

        def _solve(self, known_terms):
            self.iterations = self._solver.solve(
                                              self._terms_with_unknown,
                                              self._potential_function.vector(),
                                              known_terms)


# class FunctionManager(object):
#     """
#     TODO: Rewrite to:
#     - use INI configuration with mesh name and degree,
#     - use lazy mesh loading.
#     """
#     def __init__(self, mesh, degree):
#         with XDMFFile(mesh + '.xdmf') as fh:
#             self._mesh = Mesh()
#             fh.read(self._mesh)
#
#         self._V = FunctionSpace(self._mesh, "CG", degree)
#
#     def write(self, filename, function, name):
#         with HDF5File(MPI.comm_self,
#                       filename,
#                       'a' if os.path.exists(filename) else 'w') as fh:
#             fh.write(function, name)
#
#     def read(self, filename, name):
#         function = Function(self._V)
#         with HDF5File(MPI.comm_self, filename, 'a') as fh:
#             fh.read(function, name)
#
#         return function


class FunctionManager(object):
    function_name = 'potential'

    def __init__(self, mesh, degree=None, element_type='CG'):
        self._mesh_filename = mesh
        self._degree = degree
        self.element_type = element_type

    @property
    def degree(self):
        return self._degree

    @degree.setter
    def degree(self, value):
        self._set_degree(value)

    def _set_degree(self, value):
        if self._degree != value:
            self._degree = value
            self._delete_function_space()

    def _delete_function_space(self):
        try:
            del self._function_space
        except AttributeError:
            pass

    @property
    def mesh(self):
        try:
            return self._mesh

        except AttributeError:
            self._load_mesh()
            return self._mesh

    def _load_mesh(self):
        with XDMFFile(self._mesh_filename) as fh:
            self._mesh = Mesh()
            fh.read(self._mesh)

    @property
    def function_space(self):
        try:
            return self._function_space

        except AttributeError:
            self._create_function_space()
            return self._function_space

    def _create_function_space(self):
        logger.debug('Creating function space...')
        self._function_space = FunctionSpace(self.mesh, self.element_type, self._degree)
        logger.debug('Done.')

    def store(self, filename, function):
        with HDF5File(MPI.comm_self, filename, 'w') as fh:
            fh.write(function, self.function_name)

    def load(self, filename):
        function = self.function()
        with HDF5File(MPI.comm_self, filename, 'r') as fh:
            fh.read(function, self.function_name)

        return function

    def function(self):
        return Function(self.function_space)

    def test_function(self):
        return TestFunction(self.function_space)

    def trial_function(self):
        return TrialFunction(self.function_space)


class FunctionManagerINI(FunctionManager):
    def __init__(self, config):
        self._load_config(config)
        super(FunctionManagerINI,
              self).__init__(self.getpath('fem', 'mesh'),
                             self.getint('fem', 'degree'),
                             self.get('fem', 'element_type'))

    def getpath(self, section, field):
        return self._absolute_path(self.get(section, field))

    def _absolute_path(self, relative_path):
        return os.path.join(DIRNAME,
                            relative_path)

    def _load_config(self, config):
        self.config = configparser.ConfigParser()
        self.config.read(config)

    def load(self, name):
        return super(FunctionManagerINI,
                     self).load(self._function_filename(name))

    def _function_filename(self, name):
        return self.getpath(name, 'filename')

    def get(self, section, field):
        return self.config.get(section, field)

    def getint(self, section, field):
        return self.config.getint(section, field)

    def getfloat(self, section, field):
        return self.config.getfloat(section, field)

    def set(self, section, field, value):
        value = value if isinstance(value, str) else repr(value)

        try:
            return self.config.set(section, field, value)

        except configparser.NoSectionError:
            self.config.add_section(section)
            return self.config.set(section, field, value)

    def store(self, name, function, metadata):
        for key, value in metadata.items():
            self.set(name, key, value)

        return super(FunctionManagerINI,
                     self).store(self._function_filename(name),
                                 function)

    def write(self, filename):
        self.config.write(open(filename, 'w'))

    def functions(self):
        for section in self.config.sections():
            if section != 'fem':
                yield section

    def _set_degree(self, value):
        self.set('fem', 'degree', value)

    def has_solution(self, name):
        return self.config.has_section(name)


class PointSourceFactoryINI(object):
    def __init__(self, config):
        self._fm = FunctionManagerINI(config)

    def sources(self):
        yield from self._fm.functions()

    def __call__(self, name):
        return self.Source(self._fm.getfloat(name, 'x'),
                           self._fm.getfloat(name, 'y'),
                           self._fm.getfloat(name, 'z'),
                           conductivity=self._fm.getfloat(name, 'base_conductivity'),
                           potential_correction=self._fm.load(name))

    class Source(object):  # duplicates code from _common_new.PointSource
        def __init__(self, x, y, z, conductivity=1, amplitude=1, potential_correction=None):
            self.x = x
            self.y = y
            self.z = z
            self.conductivity = conductivity
            self.potential_correction = potential_correction
            self.a = amplitude * 0.25 / (np.pi * conductivity)

        def potential(self, X, Y, Z):
            return (self.a / np.sqrt(np.square(X - self.x)
                                     + np.square(Y - self.y)
                                     + np.square(Z - self.z))
                    + self.potential_correction(X, Y, Z))


# TODO:
# Create Romberg Function manager/controler and Romberg function factory.


if __name__ == '__main__':
    import sys

    logging.basicConfig(level=logging.INFO)

    for config in sys.argv[1:]:
        fem = SlicePointSourcePotentialFEM(config)
        solution_filename_pattern = fem._fm.get('fem', 'solution_filename_pattern')
        solution_name_pattern = fem._fm.get('fem', 'solution_name_pattern')
        solution_metadata_filename = fem._fm.getpath('fem', 'solution_metadata_filename')
        h = fem.config.getfloat('slice', 'thickness')
        k = fem._fm.getint('fem', 'k')
        n = 2 ** k + 1
        margin = 0.5 * h / n
        Z = np.linspace(margin, h - margin, n)
        X = np.linspace(0, 0.5 * h - margin, 2 ** (k - 1) + 1)

        for x_idx, x in enumerate(X):
            for y_idx, y in enumerate(X[:x_idx + 1]):
                logger.info('{:3.1f}%\t(x = {:g}\ty = {:g})'.format(100 * float(x_idx * (x_idx - 1) // 2 + y_idx) / ((2 ** (k - 1) + 1) * 2 ** (k - 2)),
                                                                    x, y))
                for z_idx, z in enumerate(Z):
                    name = solution_name_pattern.format(x=x_idx,
                                                        y=y_idx,
                                                        z=z_idx)
                    if fem._fm.has_solution(name):
                        filename = fem._fm.getpath(name, 'filename')
                        if os.path.exists(filename):
                            logger.info('{:3.1f}%\t(x = {:g}\ty = {:g},\tz={:g}) found'.format(
                                100 * float(x_idx * (x_idx - 1) // 2 + y_idx) / ((2 ** (k - 1) + 1) * 2 ** (k - 2)),
                                x, y, z))
                            continue

                    logger.info('{:3.1f}%\t(x = {:g}\ty = {:g},\tz={:g})'.format(
                        100 * float(x_idx * (x_idx - 1) // 2 + y_idx) / ((2 ** (k - 1) + 1) * 2 ** (k - 2)),
                        x, y, z))
                    function = fem.solve(x, y, z)
                    filename = solution_filename_pattern.format(x=x_idx,
                                                                y=y_idx,
                                                                z=z_idx)
                    fem._fm.store(name, function,
                                  {'filename': filename,
                                   'x': x,
                                   'y': y,
                                   'z': z,
                                   'global_preprocessing_time': float(fem.global_preprocessing_time),
                                   'local_preprocessing_time': float(fem.local_preprocessing_time),
                                   'solving_time': float(fem.solving_time),
                                   'base_conductivity': fem.base_conductivity(x, y, z),
                                   })
                fem._fm.write(solution_metadata_filename)

