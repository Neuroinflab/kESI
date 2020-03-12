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


def empty_array(shape):
    A = np.empty(shape)
    A.fill(np.nan)
    return A


class _SomeSphereGaussianLoaderBase(object):
    ATTRIBUTES = ['k',
                  'source_resolution',
                  'cortex_radius_internal',
                  'cortex_radius_external',
                  'brain_conductivity',
                  'brain_radius',
                  'degree',
                  'POTENTIAL',
                  'A',
                  # 'STATS',
                  ]

    def _load(self):
        self._populate_solutions()

        span = self.cortex_radius_external - self.cortex_radius_internal
        n = 2 ** self.k
        sd = span / n

        # computable
        self.standard_deviation = sd
        self.R = np.linspace(self.cortex_radius_internal + sd / 2 / self.source_resolution,
                             self.cortex_radius_external - sd / 2 / self.source_resolution,
                             n * self.source_resolution)

    def _load_solutions(self):
        with np.load(self.path) as fh:
            for attr in self.ATTRIBUTES:
                setattr(self, attr, fh[attr])

            self.STATS = list(fh['STATS'])


class _FixedElectrodesGaussianLoaderBase(_SomeSphereGaussianLoaderBase):
    ATTRIBUTES = _SomeSphereGaussianLoaderBase.ATTRIBUTES + ['ELECTRODES']
    _REGISTRATION_RADIUS = 0.079
    _RADIUS = 73.21604532
    _CENTER = np.array([[82.40997559, 118.14496578, 104.73314426]])
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
    ELECTRODES = (_COORDS_NW_LATER - _CENTER) / _RADIUS * _REGISTRATION_RADIUS

    def _load(self):
        super(_FixedElectrodesGaussianLoaderBase, self)._load()
        sd = self.standard_deviation

        self.ALTITUDE = []
        self.AZIMUTH = []
        for i, altitude in enumerate(
                             np.linspace(
                                  0,
                                  np.pi / 2,
                                  int(np.ceil(self.source_resolution * self.cortex_radius_external * np.pi / 2 / sd)) + 1)):
            for azimuth in np.linspace(0 if i % 2 else 2 * np.pi,
                                       2 * np.pi if i % 2 else 0,
                                       int(np.ceil(self.source_resolution * self.cortex_radius_external * np.cos(altitude) * np.pi * 2 / sd)) + 1)[:-1]:
                self.ALTITUDE.append(altitude)
                self.AZIMUTH.append(azimuth)


class _GaussianLoaderBase(_SomeSphereGaussianLoaderBase):
    ATTRIBUTES = _SomeSphereGaussianLoaderBase.ATTRIBUTES +\
                 ['scalp_radius',
                  'sampling_frequency',
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
        for x_idx in range(self.sampling_frequency + 1):
            for z_idx in range(x_idx + 1):
                assert _idx == x_idx * (x_idx + 1) // 2 + z_idx
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


class _SourceBase(object):
    def csd(self, X, Y, Z):
        return np.where((X**2 + Y**2 + Z**2 > self.parent.brain_radius ** 2),
                        0,
                        self._a
                        * np.exp(-0.5
                                 * (np.square(X - self._x)
                                    + np.square(Y - self._y)
                                    + np.square(Z - self._z))
                                 / self.parent.standard_deviation ** 2))

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def r(self):
        return self._r

    @property
    def altitude(self):
        return self._altitude

    @property
    def azimuth(self):
        return self._azimuth


class _RotatingSourceBase(_SourceBase):
    def __init__(self, r, altitude, azimuth, a, interpolator, parent):
        self._r = r
        self._altitude = altitude
        self._azimuth = azimuth

        sin_alt = np.sin(altitude)  # np.cos(np.pi / 2 - altitude)
        cos_alt = np.cos(altitude)  # np.sin(np.pi / 2 - altitude)
        sin_az = np.sin(azimuth)  # -np.sin(-azimuth)
        cos_az = np.cos(azimuth)  # np.cos(-azimuth)

        self._ROT = np.matmul([[sin_alt, cos_alt, 0],
                               [-cos_alt, sin_alt, 0],
                               [0, 0, 1]],
                              [[cos_az, 0, sin_az],
                               [0, 1, 0],
                               [-sin_az, 0, cos_az]]
                              )

        r2 = r * cos_alt
        self._x = r2 * cos_az
        self._y = r * sin_alt
        self._z = -r2 * sin_az
        self._a = a
        self.parent = parent

        self._interpolator = interpolator

    def _rotated(self, X, Y, Z):
        _X = self._ROT[0, 0] * X + self._ROT[1, 0] * Y + self._ROT[2, 0] * Z
        _Y = self._ROT[0, 1] * X + self._ROT[1, 1] * Y + self._ROT[2, 1] * Z
        _Z = self._ROT[0, 2] * X + self._ROT[1, 2] * Y + self._ROT[2, 2] * Z
        return _X, _Y, _Z


class SomeSphereFixedElectrodesGaussianSourceFactory(_FixedElectrodesGaussianLoaderBase):
    def __init__(self, filename):
        self.path = filename
        self._load()
        self._r_index = {r: i for i, r in enumerate(self.R)}
        self._altitude_azimuth_index = {coords: i
                                        for i, coords
                                        in enumerate(zip(self.ALTITUDE,
                                                         self.AZIMUTH))}

    def _populate_solutions(self):
        self._load_solutions()

    def __call__(self, r, altitude, azimuth):
        i_r = self._r_index[r]
        i_aa = self._altitude_azimuth_index[altitude, azimuth]
        POTENTIAL = self.POTENTIAL[i_r, i_aa]
        a = self.A[i_r]
        return self._Source(r, altitude, azimuth, a, POTENTIAL, self)

    class _Source(_SourceBase):
        def __init__(self, r, altitude, azimuth, a, POTENTIAL, parent):
            self._r = r
            self._altitude = altitude
            self._azimuth = azimuth
            r2 = r * np.cos(altitude)
            self._x = r2 * np.cos(azimuth)
            self._y = r * np.sin(altitude)
            self._z = -r2 * np.sin(azimuth)
            self._a = a
            self.parent = parent
            self._POTENTIAL = POTENTIAL

        def potential(self):
            return self._POTENTIAL


class _SomeSphereGaussianSourceFactoryBase(object):
    def __init__(self, filename):
        self.path = filename
        self._load()
        self._r_index = {r: i for i, r in enumerate(self.R)}
        self._interpolator = [None] * len(self.R)

    def _populate_solutions(self):
        self._load_solutions()

    def __call__(self, r, altitude, azimuth):
        a, interpolator = self._source_prefabricates(r)
        return self._Source(r, altitude, azimuth, a,
                            interpolator,
                            self)

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
        POTENTIAL = empty_array((sf + 1,
                                 len(self.Y_SAMPLE),
                                 sf + 1))
        for xz_idx, (x_idx, z_idx) in enumerate(self._xz_idx):
            P = COMPRESSED[xz_idx, :]
            POTENTIAL[x_idx, :, z_idx] = P
            if x_idx > z_idx:
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


class SomeSphereGaussianSourceFactory2D(_SomeSphereGaussianSourceFactoryBase,
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


class _SomeSphereControllerBase(object):
    FEM_ATTRIBUTES = ['brain_conductivity',
                      'brain_radius',
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
        np.savez_compressed(path, **results)

    def save2D(self, path):
        results = self._results()
        results['STATS'] = self.STATS
        results['POTENTIAL'] = self.POTENTIAL_2D
        results['sampling_frequency'] = self.sampling_frequency_2D
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

    @property
    def path(self):
        fn = '{0._fem.mesh_name}_gaussian_{1:04d}_deg_{0.degree}.npz'.format(
                   self,
                   int(round(1000 / 2 ** self.k)))

        return _fem_common._SourceFactory_Base.solution_path(fn, False)

    def _populate_solutions(self):
        try:
            self._load_solutions()

        except Exception as e:
            logger.warning(str(e))
            for attr, value in self._fem_attributes:
                setattr(self, attr, value)

            self._empty_solutions()

        else:
            self._validate_attributes()

    def _empty_solutions(self):
        n = 2 ** self.k
        self.A = empty_array(n * self.source_resolution)

    @property
    def _fem_attributes(self):
        for attr in self.FEM_ATTRIBUTES:
            logger.debug(attr)
            yield attr, getattr(self._fem, attr)


class _SomeSphereFixedElectrodesGaussianController(
          _FixedElectrodesGaussianLoaderBase,
          _SomeSphereControllerBase):
    def _empty_solutions(self):
        super(_SomeSphereFixedElectrodesGaussianController,
              self)._empty_solutions()
        n = 2 ** self.k
        self.STATS = []
        self.POTENTIAL = empty_array((n * self.source_resolution,
                                      len(self.AZIMUTH),
                                      len(self.ELECTRODES)))


class _SomeSphereGaussianController3D(_GaussianLoaderBase3D,
                                      _SomeSphereControllerBase):
    FEM_ATTRIBUTES = _SomeSphereControllerBase.FEM_ATTRIBUTES + ['scalp_radius']

    sampling_frequency = 256
    sampling_frequency_2D = 1024

    def _empty_solutions(self):
        super(_SomeSphereGaussianController3D, self)._empty_solutions()
        n = 2 ** self.k
        self.STATS = []
        xz_size = (self.sampling_frequency + 1) * (self.sampling_frequency + 2) // 2
        self.POTENTIAL = empty_array((n * self.source_resolution,
                                      xz_size,
                                      2 * self.sampling_frequency + 1))

        self.POTENTIAL_2D = empty_array((n * self.source_resolution,
                                         self.sampling_frequency_2D + 1,
                                         2 * self.sampling_frequency_2D + 1))


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
        class _SphericalGaussianPotential(_fem_common._FEM_Base):
            FRACTION_OF_SPACE = 1.0
            scalp_radius = 0.090

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


        logging.basicConfig(level=logging.INFO)

        if not os.path.exists(_fem_common.SOLUTION_DIRECTORY):
            os.makedirs(_fem_common.SOLUTION_DIRECTORY)

        for mesh_name in sys.argv[1:]:
            for SphereGaussianFEM in [OneSphereGaussianPotentialFEM,
                                      TwoSpheresGaussianPotentialFEM,
                                      FourSpheresGaussianPotentialFEM,
                                      EighthWedgeOfOneSphereGaussianPotentialFEM,
                                      EighthWedgeOfTwoSpheresGaussianPotentialFEM,
                                      EighthWedgeOfFourSpheresGaussianPotentialFEM,
                                      ]:
                if mesh_name.startswith(SphereGaussianFEM.startswith):
                    fem = SphereGaussianFEM(mesh_name=mesh_name)
                    break
            else:
                logger.warning('Missing appropriate FEM class for {}'.format(mesh_name))
                continue

            controller = _SomeSphereGaussianController3D(fem)

            for controller.degree in [1, 2, 3]:
                K_MAX = 4  # as element size is 0.25 mm,
                           # the smallest sd considered safe is
                           # 12mm / (2 ** 4)
                for controller.k in range(K_MAX + 1):
                    with controller:
                        logger.info('Gaussian SD={} ({}; deg={})'.format(
                            controller.standard_deviation,
                            mesh_name,
                            controller.degree))

                        tmp_mark = 0
                        stats = controller.STATS

                        POTENTIAL = controller.POTENTIAL
                        AS = controller.A

                        save_stopwatch = _fem_common.Stopwatch()
                        sample_stopwatch = _fem_common.Stopwatch()

                        with _fem_common.Stopwatch() as unsaved_time:
                            for idx_r, src_r in enumerate(controller.R):
                                logger.info(
                                    'Gaussian SD={}, r={} ({}, deg={})'.format(
                                        controller.standard_deviation,
                                        src_r,
                                        mesh_name,
                                        controller.degree))
                                if not np.isnan(AS[idx_r]):
                                    logger.info('Already found, skipping')
                                    continue

                                potential = controller.fem(src_r)

                                if potential is not None:
                                    with sample_stopwatch:
                                        # for idx_polar, (altitude, azimuth) in enumerate(zip(controller.ALTITUDE,
                                        #                                                     controller.AZIMUTH)):
                                        #     negative_d_altitude = np.pi / 2 - altitude
                                        #     sin_alt = np.sin(negative_d_altitude)
                                        #     cos_alt = np.cos(negative_d_altitude)
                                        #     sin_az = np.sin(-azimuth)
                                        #     cos_az = np.cos(-azimuth)
                                        #     ELECTRODES = np.matmul(
                                        #         controller.ELECTRODES,
                                        #         np.matmul(
                                        #             [[cos_alt, sin_alt, 0],
                                        #              [-sin_alt, cos_alt, 0],
                                        #              [0, 0, 1]],
                                        #             [[cos_az, 0, -sin_az],
                                        #              [0, 1, 0],
                                        #              [sin_az, 0, cos_az]]
                                        #             ))
                                        r2 = controller.scalp_radius ** 2
                                        for idx_xz, (x, z) in enumerate(controller._xz):
                                            r_xz_2 = x ** 2 + z ** 2
                                            if r_xz_2 > r2:
                                                continue
                                            for idx_y, y in enumerate(controller.Y_SAMPLE):
                                                if r_xz_2 + y ** 2 > r2:
                                                    continue
                                                try:
                                                    v = potential(x, y, z)
                                                except Exception as e:
                                                    pass
                                                else:
                                                    POTENTIAL[idx_r,
                                                              idx_xz,
                                                              idx_y] = v

                                        if fem.FRACTION_OF_SPACE < 1:
                                            logging.info('Sampling 2D')
                                            POTENTIAL_2D = controller.POTENTIAL_2D
                                            angle = fem.FRACTION_OF_SPACE * np.pi
                                            SIN_COS = np.array([np.sin(angle),
                                                                np.cos(angle)])

                                            for idx_xz, xz in enumerate(
                                                    np.linspace(0,
                                                                controller.scalp_radius,
                                                                controller.sampling_frequency_2D + 1)):
                                                x, z = SIN_COS * xz
                                                xz2 = xz ** 2
                                                for idx_y, y in enumerate(np.linspace(-controller.scalp_radius,
                                                                                      controller.scalp_radius,
                                                                                      2 * controller.sampling_frequency_2D + 1)):
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
                                AS[idx_r] = fem.a
                                stats.append((src_r,
                                              np.nan if potential is None else float(sample_stopwatch),
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
                                        sampling=sample_stopwatch.duration))

                                if float(unsaved_time) > 10 * float(save_stopwatch):
                                    with save_stopwatch:
                                        controller.save(controller.path + str(tmp_mark))
                                        if fem.FRACTION_OF_SPACE < 1:
                                            logging.info('Saving 2D')
                                            controller.save2D(controller.path.replace('.npz', '_2D.npz') + str(tmp_mark))
                                    unsaved_time.reset()
                                    tmp_mark = 1 - tmp_mark

                        if fem.FRACTION_OF_SPACE < 1:
                            controller.save2D(controller.path.replace('.npz', '_2D.npz'))
