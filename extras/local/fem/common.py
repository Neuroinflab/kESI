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
import itertools

import numpy as np
from scipy.integrate import romb
from scipy.interpolate import RegularGridInterpolator

from kesi._engine import deprecated
from kesi._verbose import VerboseFFR

try:
    from . import _common as fc
    # When run as script raises:
    #  - `ImportError` (Python 3.6-9), or
    #  - `SystemError` (Python 3.3-5), or
    #  - `ValueError` (Python 2.7).

except ImportError:
    import _common as fc


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


_DIRECTORY = os.path.dirname(__file__)


try:
    from dolfin import (assemble, Constant, cpp, Expression,
                        # FacetNormal,
                        Function, FunctionSpace, grad, HDF5File,
                        inner, KrylovSolver,
                        Measure, Mesh, MeshValueCollection, MPI,
                        TestFunction, TrialFunction, XDMFFile)

except (ModuleNotFoundError, ImportError):
    logger.warning("Unable to import from dolfin")

else:
    class LegacyConfigParser(object):
        def __init__(self, config):
            self._load_config(config)

        def getpath(self, section, field):
            return self._absolute_path(self.get(section, field))

        def _absolute_path(self, relative_path):
            return os.path.join(_DIRECTORY,
                                relative_path)

        def _load_config(self, config):
            self.config = configparser.ConfigParser()
            self.config.read(config)

        def get(self, section, field):
            return self.config.get(section, field)

        def getint(self, section, field):
            return self.config.getint(section, field)

        def getfloat(self, section, field):
            return self.config.getfloat(section, field)

        def function_filename(self, name):
            directory = os.path.dirname(self.getpath('fem',
                                                     'solution_metadata_filename'))
            return os.path.join(directory,
                                self.get(name,
                                         'filename'))

    class MetadataStorage(object):
        def __init__(self, path, sections=()):
            self._path = os.path.abspath(path)
            self._directory = os.path.dirname(self._path)
            self.config = configparser.ConfigParser()
            for section in sections:
                self.add_section(section)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            with open(self._path, 'w') as fh:
                self.config.write(fh)

        def add_section(self, section):
            self.config.add_section(section)

        def setpath(self, section, field, path):
            return self.set(section, field, self.relpath(path))

        def relpath(self, path):
            return os.path.relpath(path, start=self._directory)

        def set(self, section, field, value):
            return self._set(section,
                             field,
                             value if isinstance(value, str) else repr(value))

        def _set(self, section, field, value):
            try:
                return self.config.set(section, field, value)

            except configparser.NoSectionError:
                self.add_section(section)
                return self.config.set(section, field, value)

        def setfields(self, section, fields):
            for k, v in fields.items():
                self.set(section, k, v)


    class MetadataReader(object):
        def __init__(self, path):
            self._path = os.path.abspath(path)
            self._directory = os.path.dirname(self._path)
            self.config = configparser.ConfigParser()
            self.config.read(self._path)

        def get(self, section, field):
            return self.config.get(section, field)

        def getint(self, section, field):
            return self.config.getint(section, field)

        def getfloat(self, section, field):
            return self.config.getfloat(section, field)

        def getpath(self, section, field):
            return os.path.normpath(os.path.join(self._directory,
                                                 self.get(section, field)))


    class FunctionManager(object):
        function_name = 'potential'

        def __init__(self, mesh, degree=None, element_type='CG'):
            self.mesh_file = mesh
            self._degree = degree
            self.element_type = element_type

        @property
        def degree(self):
            return self._degree

        @property
        def mesh(self):
            try:
                return self._mesh

            except AttributeError:
                self._load_mesh()
                return self._mesh

        def _load_mesh(self):
            with XDMFFile(self.mesh_file) as fh:
                self._mesh = Mesh()
                fh.read(self._mesh)

        def _load_mesh_data(self, name, dim):
            with XDMFFile(f'{self.mesh_file[:-5]}_{name}.xdmf') as fh:
                mvc = MeshValueCollection("size_t", self.mesh, dim)
                fh.read(mvc, name)
                return cpp.mesh.MeshFunctionSizet(self.mesh, mvc)

        def load_boundaries(self):
            return self._load_mesh_data('boundaries', 2)

        def load_subdomains(self):
            return self._load_mesh_data('subdomains', 3)

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
            return os.path.join(_DIRECTORY,
                                relative_path)

        def _load_config(self, config):
            self.config = configparser.ConfigParser()
            self.config.read(config)

        def load(self, name):
            return super(FunctionManagerINI,
                         self).load(self._function_filename(name))

        def _function_filename(self, name):
            directory = os.path.dirname(self.getpath('fem',
                                                     'solution_metadata_filename'))
            return os.path.join(directory,
                                self.get(name,
                                         'filename'))

        def get(self, section, field):
            return self.config.get(section, field)

        def getint(self, section, field):
            return self.config.getint(section, field)

        def getfloat(self, section, field):
            return self.config.getfloat(section, field)

        @deprecated('New function format introduced')
        def set(self, section, field, value):
            value = value if isinstance(value, str) else repr(value)

            try:
                return self.config.set(section, field, value)

            except configparser.NoSectionError:
                self.config.add_section(section)
                return self.config.set(section, field, value)

        @deprecated('New function format introduced')
        def legacy_store(self, name, function, metadata):
            for key, value in metadata.items():
                self.set(name, key, value)

            return self.store(self._function_filename(name),
                             function)

        @deprecated('New function format introduced')
        def write(self, filename):
            self.config.write(open(filename, 'w'))

        def functions(self):
            for section in self.config.sections():
                if section != 'fem':
                    yield section

        @deprecated('New function format introduced')
        def _set_degree(self, value):
            self.set('fem', 'degree', value)

        def has_solution(self, name):
            return self.config.has_section(name)


    class _SubtractionPointSourcePotentialFEM(object):
        def __init__(self, function_manager, config):
            self._fm = function_manager
            self._setup_mesh()
            self._load_config(config)
            self._create_stopwatches()
            self._global_preprocessing()

        def _create_stopwatches(self):
            self.global_preprocessing_time = fc.Stopwatch()
            self.local_preprocessing_time = fc.Stopwatch()
            self.solving_time = fc.Stopwatch()

        def _setup_mesh(self):
            self._boundaries = self._fm.load_boundaries()
            self._subdomains = self._fm.load_subdomains()
            # self._facet_normal = FacetNormal(self._fm.mesh)

        def _load_config(self, config):
            self.config = configparser.ConfigParser()
            self.config.read(config)

        @property
        def degree(self):
            return self._fm.degree

        def _global_preprocessing(self):
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
                self._base_potential_gradient_normal_expression = self._potential_gradient_normal()
                logger.debug('Done.  Creating solver...')
                self._solver = KrylovSolver("cg", "ilu")
                self._solver.parameters["maximum_iterations"] = self.MAX_ITER
                self._solver.parameters["absolute_tolerance"] = 1E-8
                logger.debug('Done.  Solver created.')

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
            return (self.config.has_option(section, 'volume')
                    and self.config.has_option(section, 'conductivity'))

        @property
        def BOUNDARY_CONDUCTIVITY(self):
            for section in self.config.sections():
                if self._is_conductive_boundary(section):
                    yield (self.config.getint(section, 'surface'),
                           self.config.getfloat(section, 'conductivity'))

        def _is_conductive_boundary(self, section):
            return (self.config.has_option(section, 'surface')
                    and self.config.has_option(section, 'conductivity'))

        def _solve(self):
            self.iterations = self._solver.solve(
                                              self._terms_with_unknown,
                                              self._potential_function.vector(),
                                              self._known_terms)

        @deprecated('Use `.correction_potential()` method instead.')
        def solve(self, x, y, z):
            return self.correction_potential(x, y, z)

        def correction_potential(self, x , y, z):
            with self.local_preprocessing_time:
                logger.debug('Creating RHS...')
                L = self._rhs(x, y, z)
                logger.debug('Done.  Assembling linear equation vector...')
                self._known_terms = assemble(L)
                logger.debug('Done.  Assembling linear equation matrix...')
                self._terms_with_unknown = assemble(self._a)
                logger.debug('Done.')
                self._modify_linear_equation(x, y, z)

            try:
                logger.debug('Solving linear equation...')
                with self.solving_time:
                    self._solve()

                logger.debug('Done.')
                return self._potential_function

            except RuntimeError as e:
                self.iterations = self.MAX_ITER
                logger.warning("Solver failed: {}".format(repr(e)))
                return None

        def _rhs(self, x, y, z):
            base_conductivity = self.base_conductivity(x, y, z)
            self._setup_expression(self._base_potential_expression,
                                   base_conductivity, x, y, z)
            self._setup_expression(self._base_potential_gradient_normal_expression,
                                   base_conductivity, x, y, z)
            # Eq. 20 at Piastra et al 2018
            return (-sum((inner((Constant(c - base_conductivity)
                                 * grad(self._base_potential_expression)),
                                grad(self._v))
                          * self._dx(x)
                          for x, c in self.CONDUCTIVITY
                          if c != base_conductivity))
                    - sum(Constant(base_conductivity)
                          # * inner(self._facet_normal,
                          #         grad(self._base_potential_expression))
                          * self._base_potential_gradient_normal_expression
                          * self._v
                          * self._ds(s)
                          for s, _ in self.BOUNDARY_CONDUCTIVITY))

        def _setup_expression(self, expression, base_conductivity, x, y, z):
            expression.conductivity = base_conductivity
            expression.src_x = x
            expression.src_y = y
            expression.src_z = z
