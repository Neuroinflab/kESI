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

import os
import logging

import numpy as np
from scipy.interpolate import RegularGridInterpolator

try:
    from . import _fem_common
    # When run as script raises:
    #  - `ModuleNotFoundError(ImportError)` (Python 3.6-7), or
    #  - `SystemError` (Python 3.3-5), or
    #  - `ValueError` (Python 2.7).

except (ImportError, SystemError, ValueError):
    import _fem_common


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


SAMPLING_FREQUENCY = 64


class _FiniteSliceGaussianDecorator(object):
    def __call__(self, obj):
        obj.standard_deviation = obj.slice_thickness / 2 ** obj.k
        obj.X = list(np.linspace(obj.standard_deviation / 2,
                                 obj.slice_thickness - obj.standard_deviation / 2,
                                 2 ** obj.k))
        obj.X_SAMPLE = np.linspace(-obj.slice_thickness,
                                   obj.slice_thickness,
                                   2 * obj.sampling_frequency + 1)
        obj.Y_SAMPLE = np.linspace(0,
                                   obj.slice_thickness,
                                   obj.sampling_frequency + 1)


class _FiniteSliceGaussianDataFileDecorator(
          _FiniteSliceGaussianDecorator):
    ATTRIBUTES = ['slice_thickness',
                  'slice_conductivity',
                  'saline_conductivity',
                  'slice_radius',

                  'sampling_frequency',
                  'k',
                  'degree',

                  'A',
                  'POTENTIAL']

    def __init__(self, path):
        self.path = path

    def __call__(self, obj):
         self._load(obj)

         super(_FiniteSliceGaussianDataFileDecorator,
               self).__call__(obj)

    def _load(self, obj):
        with np.load(self.path) as fh:
            for attr in self.ATTRIBUTES:
                # logger.debug('setting attr ' + attr)
                setattr(obj, attr, fh[attr])
                # logger.debug('done')
            self.STATS = list(fh['STATS'])


class _FailsafeFiniteSliceGaussianController(
          _FiniteSliceGaussianDataFileDecorator):
    sampling_frequency = SAMPLING_FREQUENCY

    def __init__(self, fem):
        self._fem = fem
        self.k = None

    @property
    def slice_thickness(self):
        try:
            return self.__dict__['slice_thickness']
        except KeyError:
            return self._fem.slice_thickness

    @slice_thickness.setter
    def slice_thickness(self, value):
        self.__dict__['slice_thickness'] = value

    @property
    def slice_radius(self):
        try:
            return self.__dict__['slice_radius']
        except KeyError:
            return self._fem.slice_radius

    @slice_radius.setter
    def slice_radius(self, value):
        self.__dict__['slice_radius'] = value

    @property
    def slice_conductivity(self):
        try:
            return self.__dict__['slice_conductivity']
        except KeyError:
            return self._fem.slice_conductivity

    @slice_conductivity.setter
    def slice_conductivity(self, value):
        self.__dict__['slice_conductivity'] = value

    @property
    def saline_conductivity(self):
        try:
            return self.__dict__['saline_conductivity']
        except KeyError:
            return self._fem.saline_conductivity

    @saline_conductivity.setter
    def saline_conductivity(self, value):
        self.__dict__['saline_conductivity'] = value

    @property
    def path(self):
        fn = '{0._fem.mesh_name}_gaussian_{1:04d}_deg_{0.degree}.npz'.format(
                   self,
                   int(round(1000 / 2 ** self.k)))

        return FiniteSliceGaussianSourceFactory.solution_path(fn, False)

    def _load(self, obj):
        try:
            super(_FailsafeFiniteSliceGaussianController,
                  self)._load(obj)
        except Exception as e:
            logger.warning(str(e))
            obj.STATS = []

            obj.POTENTIAL = np.empty((2 ** obj.k,
                                      2 ** obj.k * (2 ** obj.k + 1) // 2,
                                      2 * obj.sampling_frequency + 1,
                                      obj.sampling_frequency + 1,
                                      2 * obj.sampling_frequency + 1))
            obj.POTENTIAL.fill(np.nan)

            obj.A = np.empty((2 ** obj.k, 2 ** obj.k * (2 ** obj.k + 1) // 2))
            obj.A.fill(np.nan)

    @property
    def K(self):
        return self.k

    @K.setter
    def K(self, k):
        if self.k != k:
            self.k = k
            self(self)

    def fem(self, x, y, z):
        return self._fem(int(self.degree), x, y, z, self.standard_deviation)




class FiniteSliceGaussianSourceFactory(_fem_common._SourceFactory_Base):
    def __init__(self, filename=None,
                 try_local_first=True):
         decorate = _FiniteSliceGaussianDataFileDecorator(
                        self.solution_path(filename,
                                           try_local_first))
         decorate(self)
         self.slice_radius = 3.0e-3  # m

    def __call__(self, x, y, z):
        abs_x = abs(x)
        abs_z = abs(z)

        swap_axes = False
        if abs_x < abs_z:
            swap_axes = True
            abs_x, abs_z = abs_z, abs_x

        i_x = self.X.index(abs_x)
        i_z = self.X.index(abs_z)
        idx = i_x * (i_x + 1) // 2 + i_z
        idx_y = self.X.index(y)

        POTENTIAL = self.POTENTIAL[idx_y, idx, :, :, :]
        if x < 0:
            POTENTIAL = np.flip(POTENTIAL, 0)

        if z < 0:
            POTENTIAL = np.flip(POTENTIAL, 2)

        if swap_axes:
            POTENTIAL = np.swapaxes(POTENTIAL, 0, 2)

        return self._Source(x, y, z, self.A[idx_y, idx], POTENTIAL, self)

    class _Source(object):
        def __init__(self, x, y, z, a, POTENTIAL, parent):
            self._x = x
            self._y = y
            self._z = z
            self._a = a
            self.parent = parent
            self._POTENTIAL = POTENTIAL
            self._interpolator = RegularGridInterpolator((parent.X_SAMPLE,
                                                          parent.Y_SAMPLE,
                                                          parent.X_SAMPLE),
                                                         POTENTIAL,
                                                         bounds_error=True)

        def potential(self, X, Y, Z):
            return self._interpolator(np.stack((X, Y, Z),
                                               axis=-1))

        def csd(self, X, Y, Z):
            return np.where((Y < 0)
                            | (Y > self.parent.slice_thickness)
                            | (X**2 + Z**2 > self.parent.slice_radius),
                            0,
                            self._a
                            * np.exp(-0.5
                                     * (np.square(X - self._x)
                                        + np.square(Y - self._y)
                                        + np.square(Z - self._z))
                                     / self.parent.standard_deviation ** 2))


if __name__ == '__main__':
    import sys

    try:
        from dolfin import (Expression, Constant, DirichletBC, Measure,
                            inner, grad, assemble,
                            HDF5File)

    except (ModuleNotFoundError, ImportError):
        print("""Run docker first:
        $ docker run -ti --env HOST_UID=$(id -u) --env HOST_GID=$(id -g) \\
                     -v $(pwd):/home/fenics/shared:Z \\
                     -v $(pwd)/solutions:/home/fenics/shared/solutions:Z \\
                     -w /home/fenics/shared \\
                     quay.io/fenicsproject/stable
        """)
    else:
        class GaussianPotentialFEM(_fem_common._FEM_Base):
            _RADIUS = {'finite_slice': 0.3,
                       'finite_slice_small': 0.03,
                       'finite_slice_small_coarse': 0.03,
                       'finite_slice_smaller': 0.003,
                       'finite_slice_smaller_coarse': 0.003,
                       }  # m

            slice_thickness = 0.3e-3  # m
            slice_conductivity = 0.3  # S / m
            slice_radius = 3.0e-3  # m
            saline_conductivity = 1.5  # S / m

            SLICE_VOLUME = 1
            SALINE_VOLUME = 2
            EXTERNAL_SURFACE = 3
            CONDUCTIVITY = {SLICE_VOLUME: slice_conductivity,
                            SALINE_VOLUME: saline_conductivity,
                            }

            def __init__(self, mesh_name='finite_slice'):
                super(GaussianPotentialFEM, self).__init__(
                      mesh_path=os.path.join(_fem_common.DIRNAME,
                                             'meshes',
                                             mesh_name))
                self.RADIUS = self._RADIUS[mesh_name]
                # self._k = 0
                self.mesh_name = mesh_name

            def _lhs(self):
                return sum(inner(Constant(c) * grad(self._potential_trial),
                                 grad(self._v)) * self._dx(k)
                           for k, c in self.CONDUCTIVITY.items())

            def _csd_normalization_factor(self, csd):
                old_a = csd.a
                csd.a = 1
                try:
                    return 1.0 / assemble(csd * Measure("dx", self._mesh))
                finally:
                    csd.a = old_a

            def _boundary_condition(self, *args, **kwargs):
                return DirichletBC(self._V,
                                   Constant(
                                       self.potential_behind_dome(
                                           self.RADIUS,
                                           *args, **kwargs)),
                                   self._boundaries,
                                   self.EXTERNAL_SURFACE)


            def _make_csd(self, degree, x, y, z, standard_deviation):
                return Expression(f'''
                                   x[1] > {self.slice_thickness}
                                   || x[0] * x[0] + x[2] * x[2] > {self.slice_radius ** 2}
                                   ?
                                   0.0
                                   :
                                   a * exp({-0.5 / standard_deviation ** 2}
                                           * ((x[0] - {x})*(x[0] - {x})
                                              + (x[1] - {y})*(x[1] - {y})
                                              + (x[2] - {z})*(x[2] - {z})
                                              ))
                                   ''',
                                  degree=degree,
                                  a=1.0)

            def potential_behind_dome(self, radius, *args, **kwargs):
                return (0.25 / np.pi / self.CONDUCTIVITY[self.SALINE_VOLUME]
                        / radius)

            @property
            def degree(self):
                return self._degree

            # @degree.setter
            # def degree(self, value):
            #     if self._degree != value:
            #         self._change_degree(value)
            #
            # def __call__(self, x, y, z):
            #     super(GaussianPotentialFEM,
            #           self)(self._degree, x, y, z)
            #
            # @property
            # def k(self):
            #     return self._k
            #
            # @k.setter
            # def k(self, value):
            #     if self._k != value:
            #         self._k = value
            #         _FiniteSliceGaussianDecorator()(self)


        logging.basicConfig(level=logging.INFO)

        if not os.path.exists(_fem_common.SOLUTION_DIRECTORY):
            os.makedirs(_fem_common.SOLUTION_DIRECTORY)

        for mesh_name in sys.argv[1:]:
            fem = GaussianPotentialFEM(mesh_name=mesh_name)
            controller = _FailsafeFiniteSliceGaussianController(fem)

            for controller.degree in [1,
                                      2]:  # 3 causes segmentation fault
                                           # or takes 40 mins

                K_MAX = 3  # as element size is SLICE_THICKNESS / 32,
                           # the smallest sd considered safe is
                           # SLICE_THICKNESS / (2 ** 3)
                for controller.K in range(K_MAX + 1):

                    degree = controller.degree
                    k = controller.k
                    sd = controller.standard_deviation
                    X = controller.X

                    logger.info('Gaussian SD={} ({}; deg={})'.format(sd,
                                                                     mesh_name,
                                                                     degree))

                    # solution_filename = '{}_gaussian_{:04d}_deg_{}.npz'.format(
                    #                                       mesh_name,
                    #                                       int(round(1000 / 2 ** k)),
                    #                                       degree)
                    tmp_mark = 0
                    stats = controller.STATS
                    results = {'k': k,
                               'slice_thickness': controller.slice_thickness,
                               'slice_radius': controller.slice_radius,
                               'slice_conductivity': controller.slice_conductivity,
                               'saline_conductivity': controller.saline_conductivity,
                               'degree': degree,
                               'STATS': stats,
                               'sampling_frequency': controller.sampling_frequency,
                               }

                    POTENTIAL = controller.POTENTIAL
                    results['POTENTIAL'] = POTENTIAL
                    AS = controller.A
                    results['A'] = AS

                    save_stopwatch = _fem_common.Stopwatch()

                    anything_new = False
                    with _fem_common.Stopwatch() as unsaved_time:
                        for idx_y, src_y in enumerate(X):
                            for idx_x, src_x in enumerate(X):
                                for idx_z, src_z in enumerate(X[:idx_x+1]):
                                    logger.info(
                                        'Gaussian SD={}, x={}, y={}, z={} ({}, deg={})'.format(
                                            sd,
                                            src_x,
                                            src_y,
                                            src_z,
                                            mesh_name,
                                            degree))
                                    idx_xz = idx_x * (idx_x + 1) // 2 + idx_z
                                    if not np.isnan(AS[idx_y, idx_xz]):
                                        logger.info('Already found, skipping')
                                        continue

                                    anything_new = True
                                    potential = controller.fem(src_x, src_y, src_z)

                                    stats.append((src_x,
                                                  src_y,
                                                  src_z,
                                                  potential is not None,
                                                  fem.iterations,
                                                  float(fem.solving_time),
                                                  float(fem.local_preprocessing_time),
                                                  float(fem.global_preprocessing_time)))

                                    # if potential is not None:
                                    #     with HDF5File(fem._mesh.mpi_comm(),
                                    #                   GaussianSourceFactory.solution_path(
                                    #             '{}_gaussian_{:04d}_{}_{}_{}_{}.h5'.format(
                                    #                 mesh_name,
                                    #                 int(round(1000 / 2 ** k)),
                                    #                 degree,
                                    #                 idx_y,
                                    #                 idx_x,
                                    #                 idx_z),
                                    #             False),
                                    #             'w') as fh:
                                    #         fh.write(potential, 'potential')

                                    AS[idx_y, idx_xz] = fem.a
                                    if potential is not None:
                                        for i, x in enumerate(controller.X_SAMPLE):
                                            for j, y in enumerate(controller.Y_SAMPLE):
                                                for kk, z in enumerate(controller.X_SAMPLE):
                                                    POTENTIAL[idx_y,
                                                              idx_xz,
                                                              i,
                                                              j,
                                                              kk] = potential(x, y, z)
                                    logger.info('Gaussian SD={}, x={}, y={}, z={} (deg={}): {}\t({fem.iterations}, {time})'.format(
                                                sd,
                                                src_x,
                                                src_y,
                                                src_z,
                                                degree,
                                                'SUCCEED' if potential is not None else 'FAILED',
                                                fem=fem,
                                                time=fem.local_preprocessing_time.duration + fem.solving_time.duration))
                                    if float(unsaved_time) > 10 * float(save_stopwatch):
                                        with save_stopwatch:
                                            np.savez_compressed(controller.path
                                                                + str(tmp_mark),
                                                                **results)
                                        unsaved_time.reset()
                                        tmp_mark = 1 - tmp_mark

                    if anything_new:
                        np.savez_compressed(controller.path,
                                            **results)
