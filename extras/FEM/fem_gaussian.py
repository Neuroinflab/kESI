# $ docker run -ti --env HOST_UID=$(id -u) --env HOST_GID=$(id -g) -v $(pwd):/home/fenics/shared:Z quay.io/fenicsproject/stable
# fenics@...$ cd /home/fenics/shared/

import os
import logging
import itertools

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import numpy as np
from scipy.interpolate import RegularGridInterpolator


DIRNAME = os.path.dirname(__file__)
SOLUTION_DIRECTORY = os.path.join(DIRNAME,
                                  'solutions')
SAMPLING_FREQUENCY = 5

class GaussianSourceFactory(object):
    def __init__(self, filename='eighth_of_sphere_gaussian.npz',
                 degree=1,
                 _limit=np.inf,
                 ground_truth=False):
        with np.load(self.solution_path(filename)) as fh:
            try:
                sampling_frequency = fh['sampling_frequency']
            except KeyError:
                sampling_frequency = 1
            self.standard_deviation = fh['standard_deviation']

            N = min(_limit, fh['N'])

            self.a = fh['A_{}'.format(degree)]
            COMPRESSED = fh[
                'Ground_truth'
                if ground_truth else
                'Gaussian_{}'.format(degree)]

            stride = 2 * N - 1

            self.POTENTIAL = self.empty(stride ** 3)
            self.X = self.empty(stride ** 3)
            self.Y = self.empty(stride ** 3)
            self.Z = self.empty(stride ** 3)

            POTENTIAL = self.empty((stride, stride, stride))

            for x in range(0, N * sampling_frequency, sampling_frequency):
                for y in range(0, x + 1, sampling_frequency):
                    for z in range(0, y + 1, sampling_frequency):
                        val = COMPRESSED[x * (x + 1) * (x + 2) // 6
                                         + y * (y + 1) // 2
                                         + z]

                        for xs, ys, zs in itertools.permutations(
                                [[x // sampling_frequency] if x == 0 else [-x // sampling_frequency, x // sampling_frequency],
                                 [y // sampling_frequency] if y == 0 else [-y // sampling_frequency, y // sampling_frequency],
                                 [z // sampling_frequency] if z == 0 else [-z // sampling_frequency, z // sampling_frequency]]):
                            # if x == y, x == z or y == z may repeat xs, ys, zs
                            for i, j, k in itertools.product(xs, ys, zs):
                                idx = ((N - 1 + i) * stride + N - 1 + j) * stride + N - 1 + k

                                self.POTENTIAL[idx] = val
                                self.X[idx] = i
                                self.Y[idx] = j
                                self.Z[idx] = k

                                POTENTIAL[N - 1 + i,
                                          N - 1 + j,
                                          N - 1 + k] = val

        self.interpolator = RegularGridInterpolator((np.linspace(1 - N, N - 1, stride),
                                                     np.linspace(1 - N, N - 1, stride),
                                                     np.linspace(1 - N, N - 1, stride)),
                                                    POTENTIAL)

    @classmethod
    def solution_path(cls, solution_filename):
        return os.path.join(SOLUTION_DIRECTORY,
                            solution_filename)

    def empty(self, shape):
        X = np.empty(shape)
        X.fill(np.nan)
        return X

    def csd(self, X, Y, Z):
        return np.exp(-0.5
                      * (np.square(X)
                         + np.square(Y)
                         + np.square(Z))
                      / self.standard_deviation ** 2) * self.a

    # TODO: handle cases of distant points
    def potential_sinc(self, X, Y, Z):
        return np.inner((np.sinc(np.subtract.outer(X, self.X))
                         * np.sinc(np.subtract.outer(Y, self.Y))
                         * np.sinc(np.subtract.outer(Z, self.Z))),
                        self.POTENTIAL)

    def potential_linear(self, X, Y, Z):
        return self.interpolator(np.stack((X, Y, Z),
                                          axis=-1))


    class _Source(object):
        def __init__(self,
                     scale,
                     conductivity,
                     x,
                     y,
                     z,
                     parent):
            self._scale = scale
            self._conductivity = conductivity
            self._x = x
            self._y = y
            self._z = z
            self._parent = parent

        def csd(self, X, Y, Z):
            return (self._parent.csd((X - self._x) / self._scale,
                                     (Y - self._y) / self._scale,
                                     (Z - self._z) / self._scale)
                    / self._scale ** 3)

        def _normalize(self, f, X, Y, Z):
            return (f((X - self._x) / self._scale,
                      (Y - self._y) / self._scale,
                      (Z - self._z) / self._scale)
                    / (self._scale * self._conductivity))

        def potential_sinc(self, X, Y, Z):
            return self._normalize(self._parent.potential_sinc, X, Y, Z)

        def potential(self, X, Y, Z):
            return self.potential_linear(X, Y, Z)

        def potential_linear(self, X, Y, Z):
            return self._normalize(self._parent.potential_linear, X, Y, Z)

    def Source(self, x=0, y=0, z=0, standard_deviation=1, conductivity=1):
        return self._Source(standard_deviation / self.standard_deviation,
                            conductivity,
                            x, y, z,
                            self)


if __name__ == '__main__':
    import sys
    from scipy.special import erf

    try:
        from dolfin import Expression

    except (ModuleNotFoundError, ImportError):
        print("""Run docker first:
        $ docker run -ti --env HOST_UID=$(id -u) --env HOST_GID=$(id -g) -v $(pwd):/home/fenics/shared:Z quay.io/fenicsproject/stable
        $ cd /home/fenics/shared/
        """)
    else:
        from _fem_common import _SymmetricFEM_Base

        class GaussianPotentialFEM(_SymmetricFEM_Base):
            def __init__(self, degree=1, mesh_name='eighth_of_sphere'):
                super(GaussianPotentialFEM, self).__init__(
                    degree=degree,
                    mesh_path=os.path.join(DIRNAME,
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

            def _potential_behind_dome(self, radius, standard_deviation):
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

        if not os.path.exists(SOLUTION_DIRECTORY):
            os.makedirs(SOLUTION_DIRECTORY)

        mesh_name = sys.argv[1]

        fem = GaussianPotentialFEM(mesh_name=mesh_name)
        N = 1 + int(np.floor(fem.RADIUS / np.sqrt(3)))

        for sd in [1, 2, 0.5, 0.25]:
            solution_filename = '{}_gaussian_{:04d}.npz'.format(mesh_name,
                                                                int(round(1000 * sd)))
            stats = []
            results = {'N': N,
                       'standard_deviation': sd,
                       'STATS': stats,
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
                    N_LIMIT = (N - 1) * SAMPLING_FREQUENCY + 1
                    POTENTIAL = np.empty(N_LIMIT * (N_LIMIT + 1) * (N_LIMIT + 2) // 6)
                    POTENTIAL.fill(np.nan)
                    GT = POTENTIAL.copy()
                    for x in range(N_LIMIT):
                        for y in range(x + 1):
                            for z in range(y + 1):
                                idx = x * (x + 1) * (x + 2) // 6 + y * (
                                            y + 1) // 2 + z
                                POTENTIAL[idx] = potential(x / float(SAMPLING_FREQUENCY),
                                                           y / float(SAMPLING_FREQUENCY),
                                                           z / float(SAMPLING_FREQUENCY))
                                GT[idx] = ground_truth.potential(x / float(SAMPLING_FREQUENCY),
                                                                 y / float(SAMPLING_FREQUENCY),
                                                                 z / float(SAMPLING_FREQUENCY))
                    results['Gaussian_{}'.format(degree)] = POTENTIAL
                    results['Ground_truth'.format(sd)] = GT
                    results['A_{}'.format(degree)] = fem.a
                    np.savez_compressed(GaussianSourceFactory.solution_path(solution_filename),
                                        **results)
