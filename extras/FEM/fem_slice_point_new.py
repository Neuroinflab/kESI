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
import collections

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import romb

from kesi._engine import deprecated
from kesi._verbose import VerboseFFR

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


_DIRECTORY = os.path.dirname(__file__)


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
            # self._facet_normal = FacetNormal(self._fm.mesh)

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
                self._base_potential_gradient_z_expression = self._potential_gradient_z()
                logger.debug('Done.  Creating solver...')
                self._solver = KrylovSolver("cg", "ilu")
                self._solver.parameters["maximum_iterations"] = self.MAX_ITER
                self._solver.parameters["absolute_tolerance"] = 1E-8
                logger.debug('Done.  Solver created.')


        def _potential_expression(self, conductivity=0.0):
            return Expression('''
                              0.25 / {pi} / conductivity
                              / sqrt((src_x - x[0])*(src_x - x[0])
                                     + (src_y - x[1])*(src_y - x[1])
                                     + (src_z - x[2])*(src_z - x[2]))
                              '''.format(pi=np.pi),
                              degree=self.degree,
                              domain=self._fm.mesh,
                              src_x=0.0,
                              src_y=0.0,
                              src_z=0.0,
                              conductivity=conductivity)

        def _potential_gradient_z(self, conductivity=0.0):
            return Expression('''
                              0.25 / {pi} / conductivity
                              * (src_z - x[2])
                              * pow((src_x - x[0])*(src_x - x[0])
                                    + (src_y - x[1])*(src_y - x[1])
                                    + (src_z - x[2])*(src_z - x[2]),
                                    -1.5)
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
            self._setup_expression(self._base_potential_expression,
                                   base_conductivity, x, y, z)
            self._setup_expression(self._base_potential_gradient_z_expression,
                                   base_conductivity, x, y, z)
            return (-sum((inner((Constant(c - base_conductivity)
                                 * grad(self._base_potential_expression)),
                                grad(self._v))
                          * self._dx(x)
                          for x, c in self.CONDUCTIVITY
                          if c != base_conductivity))
                    # # Eq. 18 at Piastra et al 2018
                    + sum(Constant(c)
                          # * inner(self._facet_normal,
                          #         grad(self._base_potential_expression))
                          * self._base_potential_gradient_z_expression
                          * self._v
                          * self._ds(s)
                          for s, c in self.BOUNDARY_CONDUCTIVITY))

        def _setup_expression(self, expression, base_conductivity, x, y, z):
            expression.conductivity = base_conductivity
            expression.src_x = x
            expression.src_y = y
            expression.src_z = z

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


    class PointSourceFactoryBase(object):
        def __init__(self, config):
            self._fm = FunctionManagerINI(config)
            self._fm.set('fem', 'solution_metadata_filename',
                         os.path.relpath(config,
                                         _DIRECTORY))

        def getfloat(self, name, field):
            return self._fm.getfloat(name, field)

        def load(self, name):
            return self._fm.load(name)


    class PointSourceFactoryINI(PointSourceFactoryBase):
        def __iter__(self):
            for name in self._fm.functions():
                yield self(name)

        def __call__(self, name):
            return self.LazySource(self, name)

        class LazySource(object):
            __slots__ = ('_parent',
                         '_name',
                         '_potential_correction',
                         '_a')

            def __init__(self, parent, name):
                self._parent = parent
                self._name = name
                self._potential_correction = None
                self._a = 0.25 / (np.pi * self.conductivity)

            @property
            def x(self):
                return self._getfloat('x')

            @property
            def y(self):
                return self._getfloat('y')

            @property
            def z(self):
                return self._getfloat('z')

            @property
            def conductivity(self):
                return self._getfloat('base_conductivity')

            def _getfloat(self, field):
                return self._parent.getfloat(self._name, field)

            def potential(self, X, Y, Z):
                self._load_potential_correction_if_necessary()
                return (self._a / self._distance(X, Y, Z)
                        + self._potential_correction(X, Y, Z))

            def _load_potential_correction_if_necessary(self):
                if self._potential_correction is None:
                    self._load_potential_correction()

            def _load_potential_correction(self):
                self._potential_correction = BroadcastableScalarFunction(
                                                 self._parent.load(self._name))

            def _distance(self, X, Y, Z):
                return np.sqrt(np.square(X - self.x)
                               + np.square(Y - self.y)
                               + np.square(Z - self.z))


    class DecompressedPointSourceFactory(PointSourceFactoryBase):
        def __init__(self, config):
            super(DecompressedPointSourceFactory,
                  self).__init__(config)

            self._lazy_sources = {}
            self._lazy_sources_counter = collections.Counter()

        def load(self, name):
            try:
                self._lazy_sources_counter[name] += 1
                if name in self._lazy_sources:
                    return self._lazy_sources[name]

                loaded = super(DecompressedPointSourceFactory,
                               self).load(name)
                self._lazy_sources[name] = loaded
                return loaded

            except Exception as e:
                self.unload(name)
                raise e

        def unload(self, name):
            self._lazy_sources_counter[name] -= 1
            if self._lazy_sources_counter[name] == 0:
                if name in self._lazy_sources:
                    del self._lazy_sources[name]

        def __iter__(self):
            for name in self._fm.functions():
                x = self.getfloat(name, 'x')
                y = self.getfloat(name, 'y')
                z = self.getfloat(name, 'z')
                conductivity = self.getfloat(name, 'base_conductivity')
                for xx, yy in ([(x, y)]
                               if x == y else
                               [(x, y), (y, x)]):
                    for xxx in ([xx] if xx == 0 else [xx, -xx]):
                        for yyy in ([yy] if yy == 0 else [yy, -yy]):
                            yield self.Source(self,
                                              xxx,
                                              yyy,
                                              z,
                                              name,
                                              conductivity=conductivity)

        def __call__(self, x, y, z, tolerance=np.finfo(float).eps):
            abs_x = abs(x)
            abs_y = abs(y)
            flip_xy = abs_x < abs_y
            if flip_xy:
                abs_x, abs_y = abs_y, abs_x

            for name in self._fm._functions():
                if (abs(self.getfloat(name, 'x') - abs_x) < tolerance
                    and abs(self.getfloat(name, 'y') - abs_y) < tolerance
                    and abs(self.getfloat(name, 'z') - z) < tolerance):
                    return self.Source(self,
                                       x,
                                       y,
                                       z,
                                       name,
                                       conductivity=self.getfloat(name, 'base_conductivity'))

            else:
                raise self.SolutionNotFound

        class SolutionNotFound(KeyError):
            pass

        class Source(object):  # duplicates code from _common_new.PointSource
            def __init__(self, parent, x, y, z, name, conductivity=1, amplitude=1):
                self.x = x
                self.y = y
                self.z = z
                self._flip_xy = abs(x) < abs(y)
                self._wx = 1 if x > 0 else -1
                self._wy = 1 if y > 0 else -1
                self._parent = parent
                self._name = name
                self.conductivity = conductivity
                self._potential_correction = None
                self.a = amplitude * 0.25 / (np.pi * conductivity)

            def potential(self, X, Y, Z):
                return (self.a / np.sqrt(np.square(X - self.x)
                                         + np.square(Y - self.y)
                                         + np.square(Z - self.z))
                        + self.potential_correction(X, Y, Z))

            def potential_correction(self, X, Y, Z):
                if self._potential_correction is None:
                    self._potential_correction = self._parent.load(self._name)

                if self._flip_xy:
                    return self._potential_correction(self._wy * Y,
                                                      self._wx * X,
                                                      Z)
                else:
                    return self._potential_correction(self._wx * X,
                                                      self._wy * Y,
                                                      Z)

            def __del__(self):
                self._parent.unload(self._name)


class BroadcastableScalarFunction(object):
    __slots__ = ('f',)

    def __init__(self, f):
        self.f = f

    def __call__(self, *args):
        broadcast = np.broadcast(*args)
        result = np.empty(broadcast.shape)
        f = self.f
        result.flat = [f(*a) for a in broadcast]
        return result


class DegeneratedSourceBase(object):
    __slots__ = ('_parent',)

    def __init__(self, parent=None):
        self._parent = parent

    def _get_parent(self, other):
        if self._parent is not None:
            return self._parent

        return other._parent

    def __mul__(self, other):
        return DegeneratedSource(self.POTENTIAL * other,
                                 self.CSD * other,
                                 parent=self._get_parent(other))

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        if other == 0:
            return self

        return DegeneratedSource(self.POTENTIAL + other.POTENTIAL,
                                 self.CSD + other.CSD,
                                 parent=self._get_parent(other))

    def __radd__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (1. / other)

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return self - other


class DegeneratedSourcePotentialBase(DegeneratedSourceBase):
    __slots__ = ('POTENTIAL',)

    def __init__(self, potential, parent=None):
        super(DegeneratedSourcePotentialBase,
              self).__init__(parent)

        self.POTENTIAL = potential


class DegeneratedSource(DegeneratedSourcePotentialBase):
    __slots__ = ('CSD', '_csd_interpolator')

    def __init__(self, potential, csd, parent=None):
        super(DegeneratedSource,
              self).__init__(potential,
                             parent=parent)
        self.CSD = csd
        self._csd_interpolator = None

    def csd(self, X, Y, Z):
        if self._csd_interpolator is None:
            self._create_csd_interpolator(self._parent.X,
                                          self._parent.Y,
                                          self._parent.Z)

        return self._csd_interpolator(np.stack((X, Y, Z),
                                               axis=-1))

    def _create_csd_interpolator(self, X, Y, Z):
        self._csd_interpolator = RegularGridInterpolator((X,
                                                          Y,
                                                          Z),
                                                         self.CSD,
                                                         bounds_error=False)


class _LoadableObjectBase(object):
    """
    Abstract base class for loadable objects.

    Class attributes
    ----------------
    _LoadableObject__ATTRIBUTES : list
        A class attribute to be defined by subclasses to enable
        the load/save protocol.  The list contains list of attribute
        names necessary to store complete information about the object
        in a _*.npz_ file.
    """
    def __init__(self, *args, **kwargs):
        kwargs.update(zip(self._LoadableObject__ATTRIBUTES,
                          args))
        for attr, value in kwargs.items():
            setattr(self, attr, value)

    def save(self, file):
        np.savez_compressed(file,
                            **self.attribute_mapping())

    def attribute_mapping(self):
        return {attr: getattr(self, attr) for attr in self._LoadableObject__ATTRIBUTES}

    @classmethod
    def load(cls, file):
        with np.load(file) as fh:
            return cls.from_mapping(fh)

    @classmethod
    def from_mapping(cls, attributes):
        return cls(*[attributes[attr] for attr in cls._LoadableObject__ATTRIBUTES])


class LoadableGaussians3D(_LoadableObjectBase):
    _LoadableObject__ATTRIBUTES = [
        'X',
        'Y',
        'Z',
        'STANDARD_DEVIATION',
        'AMPLITUDE',
        ]

    def __init__(self, X, Y, Z, STANDARD_DEVIATION, AMPLITUDE, *args):
        super(LoadableGaussians3D,
              self).__init__(X, Y, Z, STANDARD_DEVIATION, AMPLITUDE, *args)

        self._VARIANCE = np.square(STANDARD_DEVIATION)
        self._A = AMPLITUDE * np.power(2 * np.pi * self._VARIANCE, -1.5)

    def gaussian(self, idx, X, Y, Z):
       return self._A[idx] * np.exp(-0.5 * (np.square(X - self.X[idx])
                                            + np.square(Y - self.Y[idx])
                                            + np.square(Z - self.Z[idx])) / self._VARIANCE[idx])

    class _Child(object):
        __slots__ = ('_parent', '_idx')

        def __init__(self, parent, idx):
            self._parent = parent
            self._idx = idx

        def __call__(self, X, Y, Z):
            return self._parent.gaussian(self._idx, X, Y, Z)

        @property
        def x(self):
            return self._parent.X[self._idx]

        @property
        def y(self):
            return self._parent.Y[self._idx]

        @property
        def z(self):
            return self._parent.Z[self._idx]

        @property
        def standard_deviation(self):
            return self._parent.STANDARD_DEVIATION[self._idx]

        @property
        def amplitude(self):
            return self._parent.AMPLITUDE[self._idx]

    def __iter__(self):
        for i in range(len(self.X)):
            yield self._Child(self, i)



class _DegeneratedSourcesFactoryBase(_LoadableObjectBase):
    _LoadableObject__ATTRIBUTES = [
        'X',
        'Y',
        'Z',
        'POTENTIALS',
        'ELECTRODES',
        ]

    def __init__(self, X, Y, Z, POTENTIALS, ELECTRODES):
        super(_DegeneratedSourcesFactoryBase,
              self).__init__(X, Y, Z, POTENTIALS, ELECTRODES)

        self._X, self._Y, self._Z = np.meshgrid(self.X,
                                                self.Y,
                                                self.Z,
                                                indexing='ij')

    def copy(self, electrodes=None):
        attributes = self.attribute_mapping()
        if electrodes is not None:
            attributes['ELECTRODES'] = attributes['ELECTRODES'][electrodes]
            attributes['POTENTIALS'] = attributes['POTENTIALS'][:, :, :, electrodes]

        return self.__class__.from_mapping(attributes)

    @classmethod
    def _integration_weights(cls, X):
        dx = cls._d(X)
        n = len(X)
        if 2 ** int(np.log2(n - 1)) == n - 1:
            return romb(np.eye(n), dx=dx)

        return np.full(n, dx)

    @staticmethod
    def _d(X):
        return (X.max() - X.min()) / (len(X) - 1)

    class IntegratedSource(DegeneratedSourcePotentialBase):
        __slots__ = ('csd',)

        def __init__(self, parent, potential, csd):
            super(_DegeneratedSourcesFactoryBase.IntegratedSource,
                  self).__init__(potential,
                                 parent=parent)
            self.csd = csd

        @property
        def CSD(self):
            return self.csd(self._parent._X,
                            self._parent._Y,
                            self._parent._Z)

    class _MeasurementManager(VerboseFFR.MeasurementManagerBase):
        def __init__(self, factory):
            self.number_of_measurements = factory.ELECTRODES.shape[0]

        def probe_at_single_point(self, field, electrode):
            return field.POTENTIAL[electrode]

        def probe(self, field):
            return field.POTENTIAL

    def measurement_manager(self):
        return self._MeasurementManager(self)


### BEGIN DEPRECATED ###
@deprecated('Use DegeneratedRegularSourcesFactory instead')
class DegeneratedSliceSourcesFactory(_DegeneratedSourcesFactoryBase):
    class Source(DegeneratedSourcePotentialBase):
        def __init__(self, parent, x, y, z, potential, amplitude=1):
            super(DegeneratedSliceSourcesFactory.Source,
                  self).__init__(potential,
                                 parent=parent)
            self._x = x
            self._y = y
            self._z = z
            self.amplitude = amplitude

        @property
        def CSD(self):
            parent = self._parent
            return self.amplitude * ((parent._X == self._x)
                                     & (parent._Y == self._y)
                                     & (parent._Z == self._z))

        def csd(self):
            return self.CSD

    class MemoryEfficientSource(DegeneratedSourceBase):
        __slots__ = ('_idx_x', '_idx_y', '_idx_z')

        def __init__(self, parent, x, y, z):
            self._parent = parent
            self._idx_x = x
            self._idx_y = y
            self._idx_z = z

        @property
        def x(self):
            return self._parent.X[self._idx_x]

        @property
        def y(self):
            return self._parent.Y[self._idx_y]

        @property
        def z(self):
            return self._parent.Z[self._idx_z]

        @property
        def POTENTIAL(self):
            return self._parent.POTENTIALS[self._idx_x,
                                           self._idx_y,
                                           self._idx_z,
                                           :]

        @property
        def CSD(self):
            return ((self._parent._IDX_X == self._idx_x)
                    & (self._parent._IDX_Y == self._idx_y)
                    & (self._parent._IDX_Z == self._idx_z))

    def memory_efficient_sources(self):
        for x in self._IDX_X.flatten():
            for y in self._IDX_Y.flatten():
                for z in self._IDX_Z.flatten():
                    yield self.MemoryEfficientSource(self, x, y, z)

    def __init__(self, X, Y, Z, POTENTIALS, ELECTRODES):
        super(DegeneratedSliceSourcesFactory,
              self).__init__(X, Y, Z, POTENTIALS, ELECTRODES)
        self._IDX_X = np.arange(len(X)).reshape((-1, 1, 1))
        self._IDX_Y = np.arange(len(Y)).reshape((1, -1, 1))
        self._IDX_Z = np.arange(len(Z)).reshape((1, 1, -1))


    @classmethod
    def from_factory(cls, factory, ELECTRODES, dtype=None):
        ele_z_idx = 2
        ELE_Z = ELECTRODES[:, ele_z_idx]
        n = 2 ** factory.k + 1
        midpoint = n // 2
        X = fc.empty_array(n)
        Y = fc.empty_array(n)
        Z = fc.empty_array(n)
        POTENTIALS = fc.empty_array((n, n, n, len(ELECTRODES)),
                                    dtype=dtype)

        for x_idx, y_idx, z_idx in cls._compressed_indices(factory.k):
            source = factory(factory.solution_name_pattern.format(x=x_idx,
                                                                  y=y_idx,
                                                                  z=z_idx))
            for x_idx_2, y_idx_2 in cls._decompressed_indices(x_idx, y_idx):
                wx, wy = np.sign([x_idx_2, y_idx_2])
                ele_x_idx = int(abs(x_idx_2) < abs(y_idx_2))
                ele_y_idx = 1 - ele_x_idx
                ELE_X = wx * ELECTRODES[:, ele_x_idx]
                ELE_Y = wy * ELECTRODES[:, ele_y_idx]
                try:
                    POTENTIALS[midpoint + x_idx_2,
                               midpoint + y_idx_2,
                               z_idx,
                               :] = source.potential(ELE_X, ELE_Y, ELE_Z)
                except:
                    for i, (x, y, z) in enumerate(zip(ELE_X, ELE_Y, ELE_Z)):
                        POTENTIALS[midpoint + x_idx_2,
                                   midpoint + y_idx_2,
                                   z_idx,
                                   i] = source.potential(x, y, z)

            Z[z_idx] = source.z
            X[midpoint + x_idx] = source.x
            X[midpoint - x_idx] = -source.x
            Y[midpoint + x_idx] = source.x
            Y[midpoint - x_idx] = -source.x

        return cls(X, Y, Z, POTENTIALS, ELECTRODES)

    @classmethod
    def from_reciprocal_factory(cls, factory, ELECTRODES,
                                X=None,
                                Y=None,
                                Z=None,
                                dtype=None):
        ELE_X, ELE_Y, ELE_Z = np.transpose(ELECTRODES)
        POTENTIALS = fc.empty_array((len(X), len(Y), len(Z), len(ELECTRODES)),
                                    dtype=dtype)

        ABS_ELE_X = abs(ELE_X)
        ABS_ELE_Y = abs(ELE_Y)
        FLIP_XY = ABS_ELE_Y > ABS_ELE_X
        FLIPPED_X = np.where(FLIP_XY,
                             ABS_ELE_Y,
                             ABS_ELE_X)
        FLIPPED_Y = np.where(FLIP_XY,
                             ABS_ELE_X,
                             ABS_ELE_Y)

        for name in factory:
            x, y, z = [factory.getfloat(name, field)
                       for field in ['x', 'y', 'z']]
            IDX = (FLIPPED_X == x) & (FLIPPED_Y == y) & (ELE_Z == z)
            if IDX.any():
                source = factory(name)
                for idx in np.where(IDX)[0]:
                    wx = 1 if ELE_X[idx] > 0 else -1
                    wy = 1 if ELE_Y[idx] > 0 else -1
                    ITER_X, ITER_Y = (Y * wy, X * wx) if FLIP_XY[idx] else (X * wx, Y * wy)
                    for idx_x, src_x in enumerate(ITER_X):
                        for idx_y, src_y in enumerate(ITER_Y):
                            for idx_z, src_z in enumerate(ELE_Z):
                                POTENTIALS[idx_x,
                                           idx_y,
                                           idx_z,
                                           idx] = source.potential(src_x, src_y, src_z)

        return cls(X, Y, Z, POTENTIALS, ELECTRODES)

    @classmethod
    def _compressed_indices(cls, k):
        n = 2 ** k + 1
        midpoint = n // 2
        for x in range(0, midpoint + 1):
            for y in range(0, x + 1):
                for z in range(0, n):
                    yield x, y, z

    @classmethod
    def _decompressed_indices(cls, x_idx, y_idx):
        for x_idx_2, y_idx_2 in ([(x_idx, y_idx)]
                                  if x_idx == y_idx
                                  else
                                  [(x_idx, y_idx),
                                   (y_idx, x_idx)]):
            for wx in [1, -1] if x_idx_2 else [0]:
                for wy in [1, -1] if y_idx_2 else [0]:
                    yield wx * x_idx_2, wy * y_idx_2

    def __iter__(self):
        for x_idx, x in enumerate(self.X):
            for y_idx, y in enumerate(self.Y):
                for z, POTENTIAL in zip(self.Z, self.POTENTIALS[x_idx, y_idx]):
                    yield self.Source(self, x, y, z, POTENTIAL)

    def integrated_source(self, csd):
        POTENTIAL = 0.0

        for point in itertools.product(*map(self._integration_index_weight_coord,
                                            [self.X, self.Y, self.Z])):
            indices, weights, coords = zip(*point)
            point_density = csd(*coords)
            POTENTIAL += np.product(weights) * point_density * self.POTENTIALS[indices]

        return self.IntegratedSource(self, POTENTIAL, csd)

    @classmethod
    def _integration_index_weight_coord(cls, X):
        return list(zip(itertools.count(),
                        cls._integration_weights(X),
                        X))

    class InterpolatedSource(DegeneratedSource):
        __slots__ = ()

        def __init__(self, POTENTIAL, CSD, X, Y, Z):
            super(DegeneratedSliceSourcesFactory.InterpolatedSource,
                  self).__init__(POTENTIAL, CSD)
            self._create_csd_interpolator(X, Y, Z)

        def csd(self, X, Y, Z):
            return self._csd_interpolator(np.stack((X, Y, Z),
                                                   axis=-1))

    def add_csd_interpolator(self, source):
        return self.InterpolatedSource(source.POTENTIAL,
                                       source.CSD,
                                       self.X,
                                       self.Y,
                                       self.Z)
### END DEPRECATED ###


class DegeneratedRegularSourcesFactory(_DegeneratedSourcesFactoryBase):
    def __init__(self, X, Y, Z, POTENTIALS, ELECTRODES):
        super(DegeneratedRegularSourcesFactory,
              self).__init__(X, Y, Z, POTENTIALS, ELECTRODES)
        self._IDX_X = np.arange(len(X)).reshape((-1, 1, 1))
        self._IDX_Y = np.arange(len(Y)).reshape((1, -1, 1))
        self._IDX_Z = np.arange(len(Z)).reshape((1, 1, -1))


    @classmethod
    @deprecated('Use .from_sources() method instead.')
    def from_factory(cls, factory, ELECTRODES, dtype=None):
        return cls.from_sources(factory, ELECTRODES, dtype)

    @classmethod
    def from_sources(cls, sources, ELECTRODES, dtype=None):
        sources = list(sources)

        X = set()
        Y = set()
        Z = set()

        for source in sources:
            X.add(source.x)
            Y.add(source.y)
            Z.add(source.z)

        X, Y, Z = map(sorted, [X, Y, Z])
        POTENTIALS = fc.empty_array((len(X), len(Y), len(Z), len(ELECTRODES)),
                                    dtype=dtype)

        while sources:
            source = sources.pop()
            # it is crucial not to hold reference to the source
            # to enable freeing of the loaded FEM solution

            idx_x = X.index(source.x)
            idx_y = Y.index(source.y)
            idx_z = Z.index(source.z)

            try:
                POTENTIALS[idx_x,
                           idx_y,
                           idx_z,
                           :] = source.potential(ELECTRODES[:, 0],
                                                 ELECTRODES[:, 1],
                                                 ELECTRODES[:, 2])
            except Exception:
                for idx_e, (x, y, z) in enumerate(ELECTRODES):
                    POTENTIALS[idx_x,
                               idx_y,
                               idx_z,
                               idx_e] = source.potential(x, y, z)

        return cls(X, Y, Z, POTENTIALS, ELECTRODES)

    @classmethod
    @deprecated('May be removed in the future.')
    def from_reciprocal_factory(cls, factory, ELECTRODES,
                                X=None,
                                Y=None,
                                Z=None,
                                dtype=None,
                                tolerance=np.finfo(float).eps):

        ELECTRODES = ELECTRODES.copy()
        POTENTIALS = fc.empty_array((len(X), len(Y), len(Z), len(ELECTRODES)),
                                    dtype=dtype)

        for source in factory:
            IDX = ((abs(ELECTRODES[:, 0] - source.x) < tolerance)
                   & (abs(ELECTRODES[:, 1] - source.y) < tolerance)
                   & (abs(ELECTRODES[:, 2] - source.z) < tolerance))
            if IDX.any():
                for idx in np.where(IDX)[0]:
                    ELECTRODES[idx, :] = source.x, source.y, source.z
                    for idx_x, src_x in enumerate(X):
                        for idx_y, src_y in enumerate(Y):
                            for idx_z, src_z in enumerate(Z):
                                POTENTIALS[idx_x,
                                           idx_y,
                                           idx_z,
                                           idx] = source.potential(src_x, src_y, src_z)

        return cls(X, Y, Z, POTENTIALS, ELECTRODES)

    def __iter__(self):
        for x in self._IDX_X.flatten():
            for y in self._IDX_Y.flatten():
                for z in self._IDX_Z.flatten():
                    yield self.Source(self, x, y, z)

    class Source(DegeneratedSourceBase):
        __slots__ = ('_idx_x', '_idx_y', '_idx_z')

        def __init__(self, parent, x, y, z):
            self._parent = parent
            self._idx_x = x
            self._idx_y = y
            self._idx_z = z

        @property
        def x(self):
            return self._parent.X[self._idx_x]

        @property
        def y(self):
            return self._parent.Y[self._idx_y]

        @property
        def z(self):
            return self._parent.Z[self._idx_z]

        @property
        def POTENTIAL(self):
            return self._parent.POTENTIALS[self._idx_x,
                                           self._idx_y,
                                           self._idx_z,
                                           :]

        @property
        def CSD(self):
            return ((self._parent._IDX_X == self._idx_x)
                    & (self._parent._IDX_Y == self._idx_y)
                    & (self._parent._IDX_Z == self._idx_z))


class DegeneratedIntegratedSourcesFactory(_DegeneratedSourcesFactoryBase):
    @classmethod
    def load_from_degenerated_sources_factory(cls, file):
        with np.load(file) as fh:
            attributes = {attr: fh[attr] for attr in cls._LoadableObject__ATTRIBUTES}

        POTENTIALS = attributes['POTENTIALS']
        for i, w in enumerate(cls._integration_weights(attributes['X'])):
            POTENTIALS[i, :, :, :] *= w

        for i, w in enumerate(cls._integration_weights(attributes['Y'])):
            POTENTIALS[:, i, :, :] *= w

        for i, w in enumerate(cls._integration_weights(attributes['Z'])):
            POTENTIALS[:, :, i, :] *= w

        return cls.from_mapping(attributes)

    (NO_VECTOR_INTEGRATION,
     VECTOR_INTEGRATE_Z,
     VECTOR_INTEGRATE_YZ,
     VECTOR_INTEGRATE_XYZ) = range(4)

    def _use_vector_integration_for_xyz(self):
        return self._vectorization_level == self.VECTOR_INTEGRATE_XYZ

    def _use_vector_integration_for_yz(self):
        return self._vectorization_level == self.VECTOR_INTEGRATE_YZ

    def _use_vector_integration_for_z(self):
        return self._vectorization_level == self.VECTOR_INTEGRATE_Z

    def _decrease_vectorization_level(self):
        self._vectorization_level -= 1

    def __call__(self, csd,
                 vectorization_level=VECTOR_INTEGRATE_XYZ):
        return self.IntegratedSource(self,
                                     self.integrate_potential(csd, vectorization_level),
                                     csd)

    def integrate_potential(self, csd, vectorization_level=VECTOR_INTEGRATE_XYZ):
        self._vectorization_level = vectorization_level
        self._integrate_xyz(csd)
        return self._POTENTIAL

    def _integrate_xyz(self, csd):
        if self._use_vector_integration_for_xyz():
            try:
                self._vector_integrate_xyz(csd)

            except Exception as e:
                logger.warning(f'XYZ vector integration yielded {e}')
                self._decrease_vectorization_level()

        if not self._use_vector_integration_for_xyz():
            self._scalar_integrate_x(csd)

    def _vector_integrate_xyz(self, csd):
        self._POTENTIAL = (self.POTENTIALS
                           * csd(np.reshape(self.X, (-1, 1, 1, 1)),
                                 np.reshape(self.Y, (1, -1, 1, 1)),
                                 np.reshape(self.Z, (1, 1, -1, 1)))).sum(axis=(0, 1, 2))

    def _scalar_integrate_x(self, csd):
        self._POTENTIAL = 0.0
        for idx_x, x in enumerate(self.X):
            self._integrate_yz(csd, idx_x, x)

    def _integrate_yz(self, csd, idx_x, x):
        if self._use_vector_integration_for_yz():
            try:
                self._vector_integrate_yz(csd, idx_x, x)

            except Exception as e:
                logger.warning(f'YZ vector integration yielded {e}')
                self._decrease_vectorization_level()

        if not self._use_vector_integration_for_yz():
            self._scalar_integrate_y(csd, idx_x, x)

    def _vector_integrate_yz(self, csd, idx_x, x):
        self._POTENTIAL += (self.POTENTIALS[idx_x]
                            * csd(x,
                                  np.reshape(self.Y, (-1, 1, 1)),
                                  np.reshape(self.Z, (1, -1, 1)))).sum(axis=(0, 1))

    def _scalar_integrate_y(self, csd, idx_x, x):
        for idx_y, y in enumerate(self.Y):
            self._integrate_z(csd, idx_x, idx_y, x, y)

    def _integrate_z(self, csd, idx_x, idx_y, x, y):
        if self._use_vector_integration_for_z():
            try:
                self._vector_integrate_z(csd, idx_x, idx_y, x, y)

            except Exception as e:
                logger.warning(f'Z vector integration yielded {e}')
                self._decrease_vectorization_level()

        if not self._use_vector_integration_for_z():
            self._scalar_integrate_z(csd, idx_x, idx_y, x, y)

    def _vector_integrate_z(self, csd, idx_x, idx_y, x, y):
        self._POTENTIAL += (self.POTENTIALS[idx_x, idx_y]
                            * csd(x,
                                  y,
                                  np.reshape(self.Z, (-1, 1)))).sum(axis=0)

    def _scalar_integrate_z(self, csd, idx_x, idx_y, x, y):
        for idx_z, z in enumerate(self.Z):
            self._POTENTIAL += csd(x, y, z) * self.POTENTIALS[idx_x,
                                                              idx_y,
                                                              idx_z]

    def LoadableIntegratedSourcess(self, LoadableFunctionsClass):
        class LoadableIntegratedSourcess(LoadableFunctionsClass):
            _LoadableObject__ATTRIBUTES = LoadableFunctionsClass._LoadableObject__ATTRIBUTES + ['ELECTRODES', 'POTENTIALS']

            @classmethod
            def from_factories(cls, csd_factory, integrated_sources,
                               vectorization_level=DegeneratedIntegratedSourcesFactory.VECTOR_INTEGRATE_XYZ):
                attributes = csd_factory.attribute_mapping()
                attributes['ELECTRODES'] = integrated_sources.ELECTRODES
                attributes['POTENTIALS'] = np.array([integrated_sources.integrate_potential(csd,
                                                                                            vectorization_level=vectorization_level)
                                                     for csd in csd_factory])
                return cls.from_mapping(attributes)

            class _Child(LoadableFunctionsClass._Child):
                __slots__ = ()

                def csd(self, *args, **kwargs):
                    return self(*args, **kwargs)

                @property
                def POTENTIAL(self):
                    return self._parent.POTENTIALS[self._idx]

        return LoadableIntegratedSourcess


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
                logger.info('{} {:3.1f}%\t(x = {:g}\ty = {:g})'.format(config,
                                                                       100 * float(x_idx * (x_idx - 1) // 2 + y_idx) / ((2 ** (k - 1) + 1) * 2 ** (k - 2)),
                                                                       x, y))
                for z_idx, z in enumerate(Z):
                    name = solution_name_pattern.format(x=x_idx,
                                                        y=y_idx,
                                                        z=z_idx)
                    if fem._fm.has_solution(name):
                        filename = fem._fm.getpath(name, 'filename')
                        if os.path.exists(filename):
                            logger.info('{} {:3.1f}%\t(x = {:g}\ty = {:g},\tz={:g}) found'.format(
                                config,
                                100 * float(x_idx * (x_idx - 1) // 2 + y_idx) / ((2 ** (k - 1) + 1) * 2 ** (k - 2)),
                                x, y, z))
                            continue

                    logger.info('{} {:3.1f}%\t(x = {:g}\ty = {:g},\tz={:g})'.format(
                        config,
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

