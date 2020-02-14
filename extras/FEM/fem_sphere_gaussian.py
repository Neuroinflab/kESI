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

class _SomeSphereGaussianController(object):
    _COORDS_NW_LATER = np.array([[139.667, 154.115, 82.4576],
                             [142.692, 154.864, 89.9479],
                             [144.815, 154.772, 99.6451],
                             [146.533, 154.306, 109.924],
                             [146.793, 152.317, 119.917],
                             [145.519, 149.51, 129.734],
                             [142.5, 145.33, 139.028],
                             [138.167, 139.983, 146.762],
                             [134.107, 162.626, 81.968],
                             [137.267, 163.09, 91.0938],
                             [139.0, 162.752, 101.274],
                             [140.269, 161.607, 111.542],
                             [140.621, 159.943, 121.695],
                             [139.28, 157.254, 131.404],
                             [136.56, 152.946, 140.479],
                             [132.636, 148.073, 148.438],
                             [128.5, 170.806, 82.8056],
                             [130.92, 170.867, 92.3083],
                             [132.583, 170.061, 102.457],
                             [133.44, 168.896, 112.796],
                             [133.667, 167.244, 123.129],
                             [132.519, 164.448, 133.341],
                             [130.0, 160.417, 142.392],
                             [125.76, 155.421, 150.013],
                             [122.769, 178.133, 83.3814],
                             [124.414, 177.565, 93.4052],
                             [125.429, 176.458, 103.698],
                             [126.433, 175.274, 114.01],
                             [126.435, 173.27, 124.348],
                             [125.696, 170.729, 134.461],
                             [123.5, 167.063, 143.401],
                             [119.111, 161.944, 150.995],
                             [115.286, 183.415, 84.5052],
                             [116.385, 182.716, 94.5753],
                             [117.778, 182.103, 104.61],
                             [118.5, 180.673, 115.402],
                             [118.125, 178.511, 125.278],
                             [117.25, 175.265, 135.161],
                             [115.778, 170.284, 144.184],
                             [113.409, 163.49, 151.354],
                             [106.5, 186.847, 85.5174],
                             [107.769, 186.418, 95.8093],
                             [109.304, 186.073, 105.915],
                             [109.667, 184.267, 116.196],
                             [109.143, 181.696, 126.124],
                             [108.346, 178.001, 135.869],
                             [106.455, 173.021, 144.583],
                             [104.522, 166.893, 152.495],
                             [100.471, 149.902, 112.561],
                             [103.9, 153.427, 117.047],
                             [107.062, 156.549, 121.387],
                             [109.941, 159.473, 125.435],
                             [113.077, 162.179, 129.744],
                             [115.929, 164.509, 133.653],
                             [118.2, 166.681, 137.424],
                             [120.077, 168.966, 141.202],
                             [106.7, 140.594, 112.292],
                             [107.8, 146.743, 114.049],
                             [108.588, 152.598, 115.729],
                             [109.385, 158.389, 117.228],
                             [110.0, 163.663, 118.498],
                             [110.4, 168.667, 119.319],
                             [110.0, 173.462, 120.841],
                             [109.0, 177.5, 123.229],
                             [99.4412, 139.926, 103.226],
                             [95.9286, 148.44, 106.917],
                             [92.4615, 156.715, 110.617],
                             [90.4359, 164.794, 114.143],
                             [89.8, 172.235, 117.497],
                             [91.5625, 178.643, 120.85],
                             [102.893, 152.314, 93.7946],
                             [100.125, 159.939, 88.5634],
                             [96.5769, 166.587, 83.4696],
                             [94.9565, 174.017, 78.8632],
                             [97.25, 181.778, 77.0573],
                             [102.5, 187.076, 78.8333],
                             [89.0, 173.479, 99.9167],
                             [89.3333, 172.512, 90.0116],
                             [93.8333, 172.352, 83.1684],
                             [102.125, 172.591, 75.3385],
                             [109.0, 174.658, 71.3691],
                             [118.8, 176.917, 70.4688]])
    _CENTER = np.array([[82.40997559, 118.14496578, 104.73314426]])
    _RADIUS = 73.21604532
    _REGISTRATION_RADIUS = 0.079

    ELECTRODES = (_COORDS_NW_LATER - _CENTER) / _RADIUS * _REGISTRATION_RADIUS

    cortex_radius_external = 0.079
    cortex_radius_internal = 0.067
    source_resolution = 1

    def __init__(self, fem):
        self._fem = fem
        self.k = None

    @property
    def brain_conductivity(self):
        try:
            return self.__dict__['brain_conductivity']
        except KeyError:
            return self._fem.brain_conductivity

    @property
    def brain_radius(self):
        try:
            return self.__dict__['brain_radius']
        except KeyError:
            return self._fem.brain_radius

    @property
    def path(self):
        fn = '{0._fem.mesh_name}_gaussian_{1:04d}_deg_{0.degree}.npz'.format(
                   self,
                   int(round(1000 / 2 ** self.k)))

        return _fem_common._SourceFactory_Base.solution_path(fn, False)

    @property
    def K(self):
        return self.k

    @K.setter
    def K(self, k):
        if self.k != k:
            self.k = k
            self(self)

    def __call__(self, obj):
        span = obj.cortex_radius_external - obj.cortex_radius_internal
        n = 2 ** obj.k
        sd = span / n

        # computable
        obj.standard_deviation = sd
        obj.R = np.linspace(obj.cortex_radius_internal + sd / 2 / obj.source_resolution,
                            obj.cortex_radius_external - sd / 2 / obj.source_resolution,
                            n * obj.source_resolution)

        obj.ALTITUDE = []
        obj.AZIMUTH = []
        for i, altitude in enumerate(
                             np.linspace(
                                  0,
                                  np.pi / 2,
                                  int(np.ceil(obj.source_resolution * obj.cortex_radius_external * np.pi / 2 / sd)) + 1)):
            for azimuth in np.linspace(0 if i % 2 else 2 * np.pi,
                                       2 * np.pi if i % 2 else 0,
                                       int(np.ceil(obj.source_resolution * obj.cortex_radius_external * np.cos(altitude) * np.pi * 2 / sd)) + 1)[:-1]:
                obj.ALTITUDE.append(altitude)
                obj.AZIMUTH.append(azimuth)

        # loadable
        obj.STATS = []
        obj.POTENTIAL = np.empty((n * obj.source_resolution,
                                  len(obj.AZIMUTH),
                                  len(obj.ELECTRODES)))
        obj.POTENTIAL.fill(np.nan)

        obj.A = np.empty(n * obj.source_resolution)
        obj.A.fill(np.nan)

    def fem(self, y):
        return self._fem(int(self.degree), y, self.standard_deviation)


if __name__ == '__main__':
    import sys

    try:
        from dolfin import (Expression, Constant, DirichletBC, Measure,
                            inner, grad, assemble,
                            HDF5File)

    except (ModuleNotFoundError, ImportError):
        print("""Run docker first:
        $ docker run -ti --env HOST_UID=$(id -u) --env HOST_GID=$(id -g) -v $(pwd):/home/fenics/shared:Z quay.io/fenicsproject/stable
        $ cd /home/fenics/shared/
        """)
    else:
        class _SphericalGaussianPotential(_fem_common._FEM_Base):
            def __init__(self, mesh_name='finite_slice'):
                super(_SphericalGaussianPotential, self).__init__(
                      mesh_path=os.path.join(_fem_common.DIRNAME,
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
                    return 1.0 / assemble(csd * Measure("dx", self._mesh))
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

            _ROI_VOLUME = 1
            _BRAIN_VOLUME = 2

            CONDUCTIVITY = {_ROI_VOLUME: brain_conductivity,
                            _BRAIN_VOLUME: brain_conductivity,
                            }


        class TwoSpheresGaussianPotentialFEM(_SphericalGaussianPotential):
            startswith = 'two_spheres'

            brain_conductivity = 0.33  # S / m
            skull_conductivity = brain_conductivity / 20

            brain_radius = 0.079
            # roi_radius_min = 0.067
            # roi_radius_tangent = 0.006

            _ROI_VOLUME = 1
            _BRAIN_VOLUME = 2
            _SKULL_VOLUME = 3

            CONDUCTIVITY = {_ROI_VOLUME: brain_conductivity,
                            _BRAIN_VOLUME: brain_conductivity,
                            _SKULL_VOLUME: skull_conductivity,
                            }



        logging.basicConfig(level=logging.INFO)

        if not os.path.exists(_fem_common.SOLUTION_DIRECTORY):
            os.makedirs(_fem_common.SOLUTION_DIRECTORY)

        for mesh_name in sys.argv[1:]:
            for SphereGaussianFEM in [OneSphereGaussianPotentialFEM,
                                      TwoSpheresGaussianPotentialFEM,
                                      FourSpheresGaussianPotentialFEM,
                                      ]:
                if mesh_name.startswith(SphereGaussianFEM.startswith):
                    fem = SphereGaussianFEM(mesh_name=mesh_name)
                    break
            else:
                logger.warning('Missing appropriate FEM class for {}'.format(mesh_name))
                continue

            controller = _SomeSphereGaussianController(fem)

            for controller.degree in [1, 2, 3]:
                K_MAX = 4  # as element size is 0.25 mm,
                           # the smallest sd considered safe is
                           # 12mm / (2 ** 4)
                for controller.K in range(K_MAX + 1):

                    degree = controller.degree
                    k = controller.k
                    sd = controller.standard_deviation

                    logger.info('Gaussian SD={} ({}; deg={})'.format(sd,
                                                                     mesh_name,
                                                                     degree))

                    tmp_mark = 0
                    stats = controller.STATS
                    results = {'k': k,
                               'source_resolution': controller.source_resolution,
                               'cortex_radius_internal': controller.cortex_radius_internal,
                               'cortex_radius_external': controller.cortex_radius_external,
                               'brain_conductivity': controller.brain_conductivity,
                               'brain_radius': controller.brain_radius,
                               'degree': degree,
                               'STATS': stats,
                               'ELECTRODES': controller.ELECTRODES,
                               }

                    POTENTIAL = controller.POTENTIAL
                    results['POTENTIAL'] = POTENTIAL
                    AS = controller.A
                    results['A'] = AS

                    save_stopwatch = _fem_common.Stopwatch()

                    anything_new = False
                    with _fem_common.Stopwatch() as unsaved_time:
                        for idx_r, src_r in enumerate(controller.R):
                            logger.info(
                                'Gaussian SD={}, r={} ({}, deg={})'.format(
                                    sd,
                                    src_r,
                                    mesh_name,
                                    degree))
                            if not np.isnan(AS[idx_r]):
                                logger.info('Already found, skipping')
                                continue

                            anything_new = True
                            potential = controller.fem(src_r)
                            stats.append((src_r,
                                          potential is not None,
                                          fem.iterations,
                                          float(fem.solving_time),
                                          float(fem.local_preprocessing_time),
                                          float(fem.global_preprocessing_time)))

                            AS[idx_r] = fem.a
                            if potential is not None:
                                for idx_polar, (altitude, azimuth) in enumerate(zip(controller.ALTITUDE,
                                                                                    controller.AZIMUTH)):
                                    # logger.info(
                                    #         'Gaussian SD={}, r={}, altitude={}, azimuth={} ({}, deg={})'.format(
                                    #             sd,
                                    #             src_r,
                                    #             altitude,
                                    #             azimuth,
                                    #             mesh_name,
                                    #             degree))
                                    negative_d_altitude = np.pi / 2 - altitude
                                    sin_alt = np.sin(negative_d_altitude)
                                    cos_alt = np.cos(negative_d_altitude)
                                    sin_az = np.sin(-azimuth)
                                    cos_az = np.cos(-azimuth)
                                    ELECTRODES = np.matmul(
                                        controller.ELECTRODES,
                                        np.matmul(
                                            [[cos_alt, sin_alt, 0],
                                             [-sin_alt, cos_alt, 0],
                                             [0, 0, 1]],
                                            [[cos_az, 0, -sin_az],
                                             [0, 1, 0],
                                             [sin_az , 0, cos_az]]
                                            ))
                                    for i, (x, y, z) in enumerate(ELECTRODES):
                                        POTENTIAL[idx_r,
                                                  idx_polar,
                                                  i] = potential(x, y, z)

                            logger.info('Gaussian SD={}, r={}, (deg={}): {}\t({fem.iterations}, {time})'.format(
                                        sd,
                                        src_r,
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
