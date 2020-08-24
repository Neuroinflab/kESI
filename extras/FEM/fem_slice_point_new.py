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

import logging
import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import romb

from kesi._engine import deprecated
from kesi._verbose import VerboseFFR

try:
    from . import _fem_common as fc
    from . import fem_common as fcn
    # When run as script raises:
    #  - `ModuleNotFoundError(ImportError)` (Python 3.6-7), or
    #  - `SystemError` (Python 3.3-5), or
    #  - `ValueError` (Python 2.7).

except (ImportError, SystemError, ValueError):
    import _fem_common as fc
    import fem_common as fcn


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


try:
    from dolfin import Constant, DirichletBC, Expression

except (ModuleNotFoundError, ImportError):
    logger.warning("Unable to import from dolfin")

else:
    class SlicePointSourcePotentialFEM(fcn._SubtractionPointSourcePotentialFEM):
        MAX_ITER = 1000

        def _potential_gradient_normal(self, conductivity=0.0):
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


class DecompressedSourcesXY(object):
    def __init__(self, sources):
        self._sources = sources

    def __iter__(self):
        for s in self._sources:
            for x, y in ([(s.x, s.y)]
                         if s.x == s.y else
                         [(s.x, s.y), (s.y, s.x)]):
                for xx in ([x] if x == 0 else [x, -x]):
                    for yy in ([y] if y == 0 else [y, -y]):
                        yield self.Source(xx, yy, s)

    class Source(object):
        __slots__ = ('_source', '_flip_xy', '_wx', '_wy')

        class IncompatibleSourceCoords(TypeError):
            pass

        def __init__(self, x, y, source):
            ax, ay = abs(x), abs(y)
            if source.x != max(ax, ay) or source.y != min(ax, ay):
                raise self.IncompatibleSourceCoords

            self._source = source
            self._flip_xy = ax < ay
            self._wx = 1 if x > 0 else -1
            self._wy = 1 if y > 0 else -1

        @property
        def x(self):
            return self._wx * (self._source.y
                               if self._flip_xy
                               else self._source.x)

        @property
        def y(self):
            return self._wy * (self._source.x
                               if self._flip_xy
                               else self._source.y)

        @property
        def z(self):
            return self._source.z

        def potential(self, X, Y, Z):
            if self._flip_xy:
                return self._source.potential(self._wy * Y,
                                              self._wx * X,
                                              Z)

            return self._source.potential(self._wx * X,
                                          self._wy * Y,
                                          Z)


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


class DegeneratedRegularSourcesFactory(_DegeneratedSourcesFactoryBase):
    def __init__(self, X, Y, Z, POTENTIALS, ELECTRODES):
        super(DegeneratedRegularSourcesFactory,
              self).__init__(X, Y, Z, POTENTIALS, ELECTRODES)
        self._IDX_X = np.arange(len(X)).reshape((-1, 1, 1))
        self._IDX_Y = np.arange(len(Y)).reshape((1, -1, 1))
        self._IDX_Z = np.arange(len(Z)).reshape((1, 1, -1))

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
    @deprecated('Use `.from_reciprocal_sources()` classmethod instead.')
    def from_reciprocal_factory(cls, factory, ELECTRODES,
                                X=None,
                                Y=None,
                                Z=None,
                                dtype=None,
                                tolerance=np.finfo(float).eps):
        return cls.from_reciprocal_sources(factory, ELECTRODES,
                                           X=X,
                                           Y=Y,
                                           Z=Z,
                                           dtype=dtype,
                                           tolerance=tolerance)

    @classmethod
    def from_reciprocal_sources(cls, sources, ELECTRODES,
                                X=None,
                                Y=None,
                                Z=None,
                                dtype=None,
                                tolerance=np.finfo(float).eps):
        ELECTRODES = ELECTRODES.copy()
        POTENTIALS = fc.empty_array((len(X), len(Y), len(Z), len(ELECTRODES)),
                                    dtype=dtype)

        for source in sources:
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

