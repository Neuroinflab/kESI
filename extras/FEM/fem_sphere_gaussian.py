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

from kesi._engine import deprecated

try:
    from . import _fem_common as fc
    from . import _fem_sphere_common as fsc
    # When run as script raises:
    #  - `ImportError` (Python 3.6-9), or
    #  - `SystemError` (Python 3.3-5), or
    #  - `ValueError` (Python 2.7).

except ImportError:
    import _fem_common as fc
    import _fem_sphere_common as fsc


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class _SomeSphereGaussianLoaderBase(object):
    ATTRIBUTES = ['k',
                  'source_resolution',
                  'cortex_radius_internal',
                  'cortex_radius_external',
                  'brain_conductivity',
                  'brain_radius',
                  'scalp_radius',
                  'degree',
                  'A',
                  # 'POTENTIAL',
                  # 'STATS',
                  ]

    def _load(self):
        self._provide_attributes()

        span = self.cortex_radius_external - self.cortex_radius_internal
        n = 2 ** self.k
        sd = span / n

        # computable
        self.standard_deviation = sd
        self.R = np.linspace(self.cortex_radius_internal + sd / 2 / self.source_resolution,
                             self.cortex_radius_external - sd / 2 / self.source_resolution,
                             n * self.source_resolution)

    def _load_attributes(self):
        with np.load(self.path) as fh:
            self._load_attributes_from_numpy_file(fh)

    def _load_attributes_from_numpy_file(self, fh):
        for attr in self.ATTRIBUTES:
            setattr(self, attr, fh[attr])


class _SomeSphereGaussianPotentialLoaderBase(_SomeSphereGaussianLoaderBase):
    def _load_attributes_from_numpy_file(self, fh):
        super(_SomeSphereGaussianPotentialLoaderBase,
              self)._load_attributes_from_numpy_file(fh)

        self.POTENTIAL = fh['POTENTIAL']
        self.STATS = list(fh['STATS'])


class _GaussianLoaderBase(_SomeSphereGaussianPotentialLoaderBase):
    ATTRIBUTES = _SomeSphereGaussianPotentialLoaderBase.ATTRIBUTES + \
                 ['sampling_frequency',
                  ]


class _GaussianLoaderBase3D(_GaussianLoaderBase):
    @property
    def _xz(self):
        sf = self.sampling_frequency
        for x_idx, z_idx in self._xz_idx:
            yield self.X_SAMPLE[sf + x_idx], self.Z_SAMPLE[sf + z_idx]

    @property
    def _xz_idx(self):
        _idx = 0
        for z_idx in range(self.sampling_frequency + 1):
            for x_idx in range(z_idx + 1):
                assert _idx == z_idx * (z_idx + 1) // 2 + x_idx
                yield x_idx, z_idx
                _idx += 1

    def _load(self):
        super(_GaussianLoaderBase3D, self)._load()

        self.X_SAMPLE = np.linspace(-self.scalp_radius,
                                    self.scalp_radius,
                                    2 * self.sampling_frequency + 1)
        self.Y_SAMPLE = self.X_SAMPLE
        self.Z_SAMPLE = self.X_SAMPLE


class _GaussianLoaderBase2D(_GaussianLoaderBase):
    def _load(self):
        super(_GaussianLoaderBase2D, self)._load()
        self.X_SAMPLE = np.linspace(0,
                                    self.scalp_radius,
                                    self.sampling_frequency + 1)
        self.Y_SAMPLE = np.linspace(-self.scalp_radius,
                                    self.scalp_radius,
                                    2 * self.sampling_frequency + 1)


class SomeSphereGaussianSourceFactoryOnlyCSD(_SomeSphereGaussianLoaderBase):
    def __init__(self, filename):
        self.path = filename
        self._load()
        self._r_index = {r: i for i, r in enumerate(self.R)}

    def __call__(self, r, altitude, azimuth):
        return _SourceBase(r, altitude, azimuth,
                           self.A[self._r_index[r]],
                           self)

    def _provide_attributes(self):
        self._load_attributes()


class _SourceBase(fsc._SourceBase):
    def __init__(self, r, altitude, azimuth, a, parent):
        super(_SourceBase, self).__init__(r, altitude, azimuth, parent)
        self._a = a

    def csd(self, X, Y, Z):
        return np.where((X**2 + Y**2 + Z**2 > self.parent.brain_radius ** 2),
                        0,
                        self._a
                        * np.exp(-0.5
                                 * (np.square(X - self._x)
                                    + np.square(Y - self._y)
                                    + np.square(Z - self._z))
                                 / self.parent.standard_deviation ** 2))



# class _RotatingSourceBase(_SourceBase, fcs._RotatingSourceBase):
class _RotatingSourceBase(fsc._RotatingSourceBase, _SourceBase):
    def __init__(self, r, altitude, azimuth, a, parent, interpolator):
        super(_RotatingSourceBase,
              self).__init__(r, altitude, azimuth, a, parent)

        self._interpolator = interpolator



class _SomeSphereGaussianSourceFactoryBase(object):
    def __init__(self, filename):
        self.path = filename
        self._load()
        self._r_index = {r: i for i, r in enumerate(self.R)}
        self._interpolator = [None] * len(self.R)

    def _provide_attributes(self):
        self._load_attributes()

    def __call__(self, r, altitude, azimuth):
        a, interpolator = self._source_prefabricates(r)
        return self._Source(r, altitude, azimuth, a,
                            self,
                            interpolator)

    def _source_prefabricates(self, r):
        r_idx = self._r_index[r]
        return self.A[r_idx], self._get_interpolator(r_idx)

    def _get_interpolator(self, r_idx):
        interpolator = self._interpolator[r_idx]
        if interpolator is None:
            interpolator = self._make_interpolator(self.POTENTIAL[r_idx, :, :])
            self._interpolator[r_idx] = interpolator
        return interpolator


class SomeSphereGaussianSourceFactory3D(_SomeSphereGaussianSourceFactoryBase,
                                        _GaussianLoaderBase3D):
    def _make_interpolator(self, COMPRESSED):
        sf = self.sampling_frequency
        POTENTIAL = fc.empty_array((sf + 1,
                                    len(self.Y_SAMPLE),
                                    sf + 1))
        for xz_idx, (x_idx, z_idx) in enumerate(self._xz_idx):
            P = COMPRESSED[xz_idx, :]
            POTENTIAL[x_idx, :, z_idx] = P
            if x_idx < z_idx:
                POTENTIAL[z_idx, :, x_idx] = P
        interpolator = RegularGridInterpolator((self.X_SAMPLE[sf:],
                                                self.Y_SAMPLE,
                                                self.Z_SAMPLE[sf:]),
                                               POTENTIAL,
                                               bounds_error=False,
                                               fill_value=np.nan)
        return interpolator

    class _Source(_RotatingSourceBase):
        def potential(self, X, Y, Z):
            _X, _Y, _Z = self._rotated(X, Y, Z)
            return self._interpolator(np.stack((abs(_X), _Y, abs(_Z)),
                                               axis=-1))


class SomeSphereGaussianSourceFactoryLinear2D(_SomeSphereGaussianSourceFactoryBase,
                                              _GaussianLoaderBase2D):
    def _make_interpolator(self, POTENTIAL):
        return RegularGridInterpolator((self.X_SAMPLE,
                                        self.Y_SAMPLE),
                                       POTENTIAL,
                                       bounds_error=False,
                                       fill_value=np.nan)

    class _Source(_RotatingSourceBase):
        def potential(self, X, Y, Z):
            _X, _Y, _Z = self._rotated(X, Y, Z)
            return self._interpolator(np.stack((np.sqrt(np.square(_X)
                                                        + np.square(_Z)),
                                                _Y),
                                               axis=-1))


@deprecated('SomeSphereGaussianSourceFactory2D is deprecated; use SomeSphereGaussianSourceFactoryLinear2D instead')
class SomeSphereGaussianSourceFactory2D(SomeSphereGaussianSourceFactoryLinear2D):
    pass


class _SomeSphereControllerBase(object):
    FEM_ATTRIBUTES = ['brain_conductivity',
                      'brain_radius',
                      'scalp_radius',
                      ]
    TOLERANCE = np.finfo(float).eps

    cortex_radius_external = 0.079
    cortex_radius_internal = 0.067
    source_resolution = 4

    def __init__(self, fem):
        self._fem = fem
        self.k = None

    def __enter__(self):
        self._load()
        self._anything_new = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._anything_new:
            self.save(self.path)

    def _results(self):
        return {attr: getattr(self, attr)
                for attr in self.ATTRIBUTES
                }

    def save(self, path):
        results = self._results()
        results['STATS'] = self.STATS
        results['POTENTIAL'] = self.POTENTIAL
        np.savez_compressed(path, **results)


    def fem(self, y):
        self._anything_new = True
        return self._fem(int(self.degree), y, self.standard_deviation)

    def _validate_attributes(self):
        for attr, value in self._fem_attributes:
            loaded = getattr(self, attr)
            err = (loaded - value) / value
            msg = "self.{0} / self._fem.{0} - 1 = {1:.2e}".format(attr, err)

            assert abs(err) < self.TOLERANCE, msg

    def _provide_attributes(self):
        try:
            self._load_attributes()

        except Exception as e:
            logger.warning(str(e))
            for attr, value in self._fem_attributes:
                setattr(self, attr, value)

            self._empty_solutions()

        else:
            self._validate_attributes()

    def _empty_solutions(self):
        n = 2 ** self.k
        self.A = fc.empty_array(n * self.source_resolution)
        self.STATS = []
        self.POTENTIAL = fc.empty_array(self._potential_size(n))

    @property
    def _fem_attributes(self):
        for attr in self.FEM_ATTRIBUTES:
            logger.debug(attr)
            yield attr, getattr(self._fem, attr)


class _SomeSphereGaussianController3D(_GaussianLoaderBase3D,
                                      _SomeSphereControllerBase):
    sampling_frequency = 256

    # def _empty_solutions(self):
    #     super(_SomeSphereGaussianController3D, self)._empty_solutions()
    #     n = 2 ** self.k

    def _potential_size(self, n):
        xz_size = (self.sampling_frequency + 1) * (self.sampling_frequency + 2) // 2
        return (n * self.source_resolution,
                xz_size,
                2 * self.sampling_frequency + 1)

    @property
    def path(self):
        fn = '{0._fem.mesh_name}_gaussian_{1:04d}_deg_{0.degree}.npz'.format(
                   self,
                   int(round(1000 / 2 ** self.k)))

        return fc._SourceFactory_Base.solution_path(fn, False)


class _SomeSphereGaussianController2D(_GaussianLoaderBase2D,
                                      _SomeSphereControllerBase):
    sampling_frequency = 1024

    def _potential_size(self, n):
        return (n * self.source_resolution,
                self.sampling_frequency + 1,
                2 * self.sampling_frequency + 1)

    @property
    def path(self):
        fn = '{0._fem.mesh_name}_gaussian_{1:04d}_deg_{0.degree}_2D.npz'.format(
                   self,
                   int(round(1000 / 2 ** self.k)))

        return fc._SourceFactory_Base.solution_path(fn, False)


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
                     -w /home/fenics/shared \\
                     quay.io/fenicsproject/stable
        """)
    else:
        class _SphericalGaussianPotential(fc._FEM_Base):
            FRACTION_OF_SPACE = 1.0
            scalp_radius = 0.090

            def __init__(self, mesh_name='finite_slice'):
                super(_SphericalGaussianPotential, self).__init__(
                      mesh_path=os.path.join(fc.DIRNAME,
                                             'meshes',
                                             mesh_name))
                self.mesh_name = mesh_name

            def _lhs(self):
                return sum(inner(Constant(c) * grad(self._potential_trial),
                                 grad(self._v)) * self._dx(k)
                           for k, c in self.CONDUCTIVITY.items())

            def _csd_normalization_factor(self, csd):
                old_a = csd.a
                csd.a = 1
                try:
                    return self.FRACTION_OF_SPACE / assemble(csd * Measure("dx", self._mesh))
                finally:
                    csd.a = old_a

            def _boundary_condition(self, *args, **kwargs):
                gdim = self._mesh.geometry().dim()
                dofs_x = self._V.tabulate_dof_coordinates().reshape((-1, gdim))
                R2 = np.square(dofs_x).sum(axis=1)
                # logger.debug('R2.min() == {}'.format(R2.min()))
                central_idx = np.argmin(R2)
                # logger.debug('R2[{}] == {}'.format(central_idx, R2[central_idx]))
                logger.debug('DBC at: {}, {}, {}'.format(*dofs_x[central_idx]))
                return DirichletBC(self._V,
                                   Constant(0),
                                   "near(x[0], {}) && near(x[1], {}) && near(x[2], {})".format(*dofs_x[central_idx]),
                                   "pointwise")

            def _make_csd(self, degree, y, standard_deviation):
                return Expression(f'''
                                   x[0] * x[0] + x[1] * x[1] + x[2] * x[2] > {self.brain_radius ** 2}
                                   ?
                                   0.0
                                   :
                                   a * exp({-0.5 / standard_deviation ** 2}
                                           * ((x[0])*(x[0])
                                              + (x[1] - {y})*(x[1] - {y})
                                              + (x[2])*(x[2])
                                              ))
                                   ''',
                                  degree=degree,
                                  a=1.0)

            @property
            def degree(self):
                return self._degree


        class OneSphereGaussianPotentialFEM(_SphericalGaussianPotential):
            startswith = 'one_sphere'

            brain_conductivity = 0.33  # S / m

            brain_radius = 0.079
            # roi_radius_min = 0.067
            # roi_radius_tangent = 0.006

            _BRAIN_VOLUME = 1

            CONDUCTIVITY = {
                            _BRAIN_VOLUME: brain_conductivity,
                            }


        class EighthWedgeOfOneSphereGaussianPotentialFEM(
                  OneSphereGaussianPotentialFEM):
            startswith = 'eighth_wedge_of_one_sphere'
            FRACTION_OF_SPACE = 0.125


        class SixthWedgeOfOneSphereGaussianPotentialFEM(
                  OneSphereGaussianPotentialFEM):
            startswith = 'sixth_wedge_of_one_sphere'
            FRACTION_OF_SPACE = 1.0 / 6


        class TwoSpheresGaussianPotentialFEM(_SphericalGaussianPotential):
            startswith = 'two_spheres'

            brain_conductivity = 0.33  # S / m
            skull_conductivity = brain_conductivity / 20

            brain_radius = 0.079
            # roi_radius_min = 0.067
            # roi_radius_tangent = 0.006

            _BRAIN_VOLUME = 1
            _SKULL_VOLUME = 2

            CONDUCTIVITY = {
                            _BRAIN_VOLUME: brain_conductivity,
                            _SKULL_VOLUME: skull_conductivity,
                            }


        class EighthWedgeOfTwoSpheresGaussianPotentialFEM(
                  TwoSpheresGaussianPotentialFEM):
            startswith = 'eighth_wedge_of_two_spheres'
            FRACTION_OF_SPACE = 0.125


        class SixthWedgeOfTwoSpheresGaussianPotentialFEM(
                  TwoSpheresGaussianPotentialFEM):
            startswith = 'sixth_wedge_of_two_spheres'
            FRACTION_OF_SPACE = 1.0 / 6


        class FourSpheresGaussianPotentialFEM(_SphericalGaussianPotential):
            startswith = 'four_spheres'

            brain_conductivity = 0.33  # S / m
            csf_conductivity = brain_conductivity * 5
            skull_conductivity = brain_conductivity / 20
            scalp_conductivity = brain_conductivity

            brain_radius = 0.079
            # roi_radius_min = 0.067
            # roi_radius_tangent = 0.006


            _BRAIN_VOLUME = 1
            _CSF_VOLUME = 2
            _SKULL_VOLUME = 3
            _SCALP_VOLUME = 4

            CONDUCTIVITY = {
                            _BRAIN_VOLUME: brain_conductivity,
                            _CSF_VOLUME: csf_conductivity,
                            _SKULL_VOLUME: skull_conductivity,
                            _SCALP_VOLUME: scalp_conductivity,
                            }


        class EighthWedgeOfFourSpheresGaussianPotentialFEM(
                  FourSpheresGaussianPotentialFEM):
            startswith = 'eighth_wedge_of_four_spheres'
            FRACTION_OF_SPACE = 0.125


        class SixthWedgeOfFourSpheresGaussianPotentialFEM(
                  FourSpheresGaussianPotentialFEM):
            startswith = 'sixth_wedge_of_four_spheres'
            FRACTION_OF_SPACE = 1.0 / 6


        logging.basicConfig(level=logging.INFO)

        if not os.path.exists(fc.SOLUTION_DIRECTORY):
            os.makedirs(fc.SOLUTION_DIRECTORY)

        for mesh_name in sys.argv[1:]:
            for SphereGaussianFEM in [OneSphereGaussianPotentialFEM,
                                      TwoSpheresGaussianPotentialFEM,
                                      FourSpheresGaussianPotentialFEM,
                                      EighthWedgeOfOneSphereGaussianPotentialFEM,
                                      EighthWedgeOfTwoSpheresGaussianPotentialFEM,
                                      EighthWedgeOfFourSpheresGaussianPotentialFEM,
                                      SixthWedgeOfOneSphereGaussianPotentialFEM,
                                      SixthWedgeOfTwoSpheresGaussianPotentialFEM,
                                      SixthWedgeOfFourSpheresGaussianPotentialFEM,
                                      ]:
                if mesh_name.startswith(SphereGaussianFEM.startswith):
                    fem = SphereGaussianFEM(mesh_name=mesh_name)
                    break
            else:
                logger.warning('Missing appropriate FEM class for {}'.format(mesh_name))
                continue

            controller = _SomeSphereGaussianController3D(fem)
            controller_2D = _SomeSphereGaussianController2D(fem)

            for controller.degree in [1, 2, 3]:
                controller_2D.degree = controller.degree
                K_MAX = 4  # as element size is 0.25 mm,
                           # the smallest sd considered safe is
                           # 12mm / (2 ** 4)
                for controller.k in range(K_MAX + 1):
                    controller_2D.k = controller.k
                    with controller:
                        logger.info('Gaussian SD={} ({}; deg={})'.format(
                            controller.standard_deviation,
                            mesh_name,
                            controller.degree))
                        with controller_2D:
                            tmp_mark = 0

                            POTENTIAL = controller.POTENTIAL
                            AS = controller.A

                            save_stopwatch = fc.Stopwatch()
                            sample_stopwatch = fc.Stopwatch()

                            with fc.Stopwatch() as unsaved_time:
                                for idx_r, src_r in enumerate(controller.R):
                                    logger.info(
                                        'Gaussian SD={}, r={} ({}, deg={})'.format(
                                            controller.standard_deviation,
                                            src_r,
                                            mesh_name,
                                            controller.degree))
                                    if not np.isnan(AS[idx_r]) and not np.isnan(controller_2D.A[idx_r]):
                                        logger.info('Already found, skipping')
                                        continue

                                    potential = controller.fem(src_r)

                                    if potential is not None:
                                        hits = 0
                                        exceptions = 0
                                        misses = 0
                                        with sample_stopwatch:
                                            r2 = controller.scalp_radius ** 2
                                            for idx_xz, (x, z) in enumerate(controller._xz):
                                                r_xz_2 = x ** 2 + z ** 2
                                                if r_xz_2 > r2:
                                                    misses += len(controller.Y_SAMPLE)
                                                    continue

                                                for idx_y, y in enumerate(controller.Y_SAMPLE):
                                                    if r_xz_2 + y ** 2 > r2:
                                                        misses += 1
                                                        continue
                                                    try:
                                                        v = potential(x, y, z)
                                                    except Exception as e:
                                                        if x < 0 or z < 0 or abs(y) > controller.scalp_radius or x > controller.scalp_radius or z > controller.scalp_radius:
                                                            logger.warning('coords out of bounding box')
                                                        exceptions += 1
                                                    else:
                                                        hits += 1
                                                        POTENTIAL[idx_r,
                                                                  idx_xz,
                                                                  idx_y] = v

                                        sampling_time_3D = float(sample_stopwatch)
                                        logger.info('H={} ({:.2f}),\tM={} ({:.2f}),\tE={} ({:.2f})'.format(hits,
                                                                                                           hits / float(hits + misses + exceptions),
                                                                                                           misses,
                                                                                                           misses / float(hits + misses + exceptions),
                                                                                                           exceptions,
                                                                                                           exceptions / float(hits + exceptions)))

                                        if fem.FRACTION_OF_SPACE < 1:
                                            with sample_stopwatch:
                                                logging.info('Sampling 2D')
                                                POTENTIAL_2D = controller_2D.POTENTIAL
                                                angle = fem.FRACTION_OF_SPACE * np.pi
                                                SIN_COS = np.array([np.sin(angle),
                                                                    np.cos(angle)])

                                                for idx_xz, xz in enumerate(
                                                        controller_2D.X_SAMPLE):
                                                    x, z = SIN_COS * xz
                                                    xz2 = xz ** 2
                                                    for idx_y, y in enumerate(
                                                            controller_2D.Y_SAMPLE):
                                                        if xz2 + y ** 2 > r2:
                                                            continue
                                                        try:
                                                            v = potential(x, y, z)
                                                        except Exception as e:
                                                            pass
                                                        else:
                                                            POTENTIAL_2D[idx_r,
                                                                         idx_xz,
                                                                         idx_y] = v

                                            sampling_time_2D = float(sample_stopwatch)

                                    if fem.FRACTION_OF_SPACE < 1:
                                        controller_2D.A[idx_r] = fem.a
                                        controller_2D.STATS.append((src_r,
                                                                    np.nan if potential is None else sampling_time_2D,
                                                                    fem.iterations,
                                                                    float(fem.solving_time),
                                                                    float(fem.local_preprocessing_time),
                                                                    float(fem.global_preprocessing_time)))

                                    AS[idx_r] = fem.a
                                    controller.STATS.append((src_r,
                                                             np.nan if potential is None else sampling_time_3D,
                                                             fem.iterations,
                                                             float(fem.solving_time),
                                                             float(fem.local_preprocessing_time),
                                                             float(fem.global_preprocessing_time)))

                                    logger.info(
                                        'Gaussian SD={}, r={}, (deg={}): {}\t({fem.iterations}, {time}, {sampling})'.format(
                                            controller.standard_deviation,
                                            src_r,
                                            controller.degree,
                                            'SUCCEED' if potential is not None else 'FAILED',
                                            fem=fem,
                                            time=fem.local_preprocessing_time.duration + fem.solving_time.duration,
                                            sampling=sampling_time_2D + sampling_time_3D))

                                    if float(unsaved_time) > 10 * float(save_stopwatch):
                                        with save_stopwatch:
                                            controller.save(controller.path + str(tmp_mark))
                                            if fem.FRACTION_OF_SPACE < 1:
                                                logging.info('Saving 2D')
                                                controller_2D.save(controller_2D.path + str(tmp_mark))
                                        unsaved_time.reset()
                                        tmp_mark = 1 - tmp_mark

                            if fem.FRACTION_OF_SPACE < 1:
                                controller_2D.save(controller_2D.path)
