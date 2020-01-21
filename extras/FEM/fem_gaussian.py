#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
#    Copyright (C) 2019 Jakub M. Dzik (Laboratory of Neuroinformatics;        #
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
from scipy.special import erf

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

# SOLUTION_DIRECTORY = _fem_common.SOLUTION_DIRECTORY

SAMPLING_FREQUENCY = 5

class GaussianSourceFactory(_fem_common._SymmetricSourceFactory_Base):
    def load_specific_attributes(self, fh):
        self.standard_deviation = fh['standard_deviation']

    def solution_array_name(self, degree, ground_truth=False):
        if ground_truth:
            return 'Ground_truth'

        return 'Gaussian_{}'.format(degree)

    def csd(self, X, Y, Z):
        return np.exp(-0.5
                      * (np.square(X)
                         + np.square(Y)
                         + np.square(Z))
                      / self.standard_deviation ** 2) * self.a

    def potential_behind_dome(self, radius):
        return (0.25
                * erf(radius
                      / (np.sqrt(2)
                         * self.standard_deviation))
                / (radius * np.pi))

    def __call__(self, x=0, y=0, z=0, standard_deviation=1, conductivity=1):
        return self._Source(standard_deviation / self.standard_deviation,
                            conductivity,
                            x, y, z,
                            self)


if __name__ == '__main__':
    import sys

    try:
        from dolfin import Expression

    except (ModuleNotFoundError, ImportError):
        print("""Run docker first:
        $ docker run -ti --env HOST_UID=$(id -u) --env HOST_GID=$(id -g) -v $(pwd):/home/fenics/shared:Z -w /home/fenics/shared quay.io/fenicsproject/stable
        """)
    else:
        class GaussianPotentialFEM(_fem_common._SymmetricFEM_Base):
            def __init__(self, mesh_name='eighth_of_sphere'):
                         super(GaussianPotentialFEM, self).__init__(
                               mesh_path=os.path.join(_fem_common.DIRNAME,
                                                      'meshes',
                                                      mesh_name))

            def _make_csd(self, degree, standard_deviation):
                return Expression(f'''
                                 a * exp({-0.5 / standard_deviation ** 2}
                                         * ((x[0])*(x[0])
                                            + (x[1])*(x[1])
                                            + (x[2])*(x[2])
                                            ))
                                 ''',
                                  degree=degree,
                                  a=1.0)

            def potential_behind_dome(self, radius, standard_deviation):
                return (0.25
                        * erf(radius
                              / (np.sqrt(2)
                                 * standard_deviation))
                        / (radius * np.pi))


### COPIED FROM _common_new.py
        class GaussianSourceBase(object):
            def __init__(self, x, y, z, standard_deviation):
                self.x = x
                self.y = y
                self.z = z
                self._variance = standard_deviation ** 2
                self._a = (2 * np.pi * self._variance) ** -1.5


        class GaussianSourceKCSD3D(GaussianSourceBase):
            _dtype = np.sqrt(0.5).__class__
            _fraction_of_erf_to_x_limit_in_0 = _dtype(2 / np.sqrt(np.pi))
            _x = _dtype(1.)
            _half = _dtype(0.5)
            _last = 2.
            _err = 1.
            while 0 < _err < _last:
                _radius_of_erf_to_x_limit_applicability = _x
                _last = _err
                _x *= _half
                _err = _fraction_of_erf_to_x_limit_in_0 - erf(_x) / _x

            def __init__(self, x, y, z, standard_deviation, conductivity):
                super(GaussianSourceKCSD3D, self).__init__(x, y, z,
                                                           standard_deviation)
                self.conductivity = conductivity
                self._b = 0.25 / (np.pi * conductivity)
                self._c = np.sqrt(0.5) / standard_deviation

            def csd(self, X, Y, Z):
                return self._a * np.exp(-0.5 * (
                            (X - self.x) ** 2 + (Y - self.y) ** 2 + (
                                Z - self.z) ** 2) / self._variance)

            def potential(self, X, Y, Z):
                R = np.sqrt(
                    (X - self.x) ** 2 + (Y - self.y) ** 2 + (Z - self.z) ** 2)
                Rc = R * self._c
                return self._b * np.where(
                    Rc >= self._radius_of_erf_to_x_limit_applicability,
                    erf(Rc) / R,
                    self._c * self._fraction_of_erf_to_x_limit_in_0)
### END OF COPIED FROM _common_new.py

        logging.basicConfig(level=logging.INFO)

        if not os.path.exists(_fem_common.SOLUTION_DIRECTORY):
            os.makedirs(_fem_common.SOLUTION_DIRECTORY)

        for mesh_name in sys.argv[1:]:
            fem = GaussianPotentialFEM(mesh_name=mesh_name)
            N = 1 + int(np.ceil(fem.RADIUS))

            for sd in [1, 2, 0.5, 0.25]:
                solution_filename = '{}_gaussian_{:04d}.npz'.format(mesh_name,
                                                                    int(round(1000 * sd)))
                stats = []
                results = {'N': N,
                           'standard_deviation': sd,
                           'STATS': stats,
                           'radius': fem.RADIUS,
                           'sampling_frequency': SAMPLING_FREQUENCY,
                           }
                for degree in [1, 2, 3]:
                    ground_truth = GaussianSourceKCSD3D(0, 0, 0, sd, 1)
                    logger.info('Gaussian SD={} (deg={})'.format(sd, degree))
                    potential = fem(degree, sd)

                    stats.append((degree,
                                  potential is not None,
                                  fem.iterations,
                                  fem.time.total_seconds()))
                    logger.info('Gaussian SD={} (deg={}): {}\t({fem.iterations}, {fem.time})'.format(
                                     sd,
                                     degree,
                                     'SUCCEED' if potential is not None else 'FAILED',
                                     fem=fem))
                    if potential is not None:
                        N_LIMIT = (N - 1) * SAMPLING_FREQUENCY + 1 # TODO: prove correctness
                        POTENTIAL = np.empty(N_LIMIT * (N_LIMIT + 1) * (N_LIMIT + 2) // 6)
                        POTENTIAL.fill(np.nan)
                        GT = POTENTIAL.copy()
                        for x in range(N_LIMIT):
                            for y in range(x + 1):
                                for z in range(y + 1):
                                    idx = x * (x + 1) * (x + 2) // 6 + y * (
                                                y + 1) // 2 + z
                                    xx = x / float(SAMPLING_FREQUENCY)
                                    yy = y / float(SAMPLING_FREQUENCY)
                                    zz = z / float(SAMPLING_FREQUENCY)
                                    r = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)
                                    if r >= fem.RADIUS:
                                        v = fem.potential_behind_dome(r, sd)
                                    else:
                                        try:
                                            v = potential(xx, yy, zz)
                                        except RuntimeError as e:
                                            logger.warning("""
                                    potential({}, {}, {})
                                    (r = {})
                                    raised:
                                    {}""".format(xx, yy, zz, r, e))
                                            v = fem.potential_behind_dome(r, sd)
                                    POTENTIAL[idx] = v
                                    GT[idx] = ground_truth.potential(xx,
                                                                     yy,
                                                                     zz)
                        results['Gaussian_{}'.format(degree)] = POTENTIAL
                        results['Ground_truth'] = GT
                        results['A_{}'.format(degree)] = fem.a
                        np.savez_compressed(GaussianSourceFactory.solution_path(
                                                solution_filename,
                                                False),
                                            **results)
