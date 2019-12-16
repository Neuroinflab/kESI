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


class LanczosSourceFactory(object):
    def __init__(self, filename='Lanczos.npz', degree=1, _limit=np.inf):
        with np.load(self.solution_path(filename)) as fh:
            sampling_frequency = fh['sampling_frequency']
            self.n = fh['folds']

            N = min(_limit, fh['N'])

            self.a = fh['A_{}'.format(degree)]
            COMPRESSED = fh['Lanczos_{}'.format(degree)]

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

    def _lanczos(self, X):
        return np.where(abs(X) >= self.n,
                        0,
                        np.sinc(X) * np.sinc(X / self.n))

    def csd(self, X, Y, Z):
        return self._lanczos(X) * self._lanczos(Y) * self._lanczos(Z) * self.a

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

    def Source(self, x=0, y=0, z=0, scale=1, conductivity=1):
        return self._Source(scale,
                            conductivity,
                            x, y, z,
                            self)



if __name__ == '__main__':
    import sys
    try:
        from dolfin import Constant, Mesh, MeshFunction, FunctionSpace, TestFunction, TrialFunction, Function, Measure, inner, grad, assemble, KrylovSolver
        from dolfin import Expression, DirichletBC

    except (ModuleNotFoundError, ImportError):
        print("""Run docker first:
        $ docker run -ti --env HOST_UID=$(id -u) --env HOST_GID=$(id -g) -v $(pwd):/home/fenics/shared:Z quay.io/fenicsproject/stable
        $ cd /home/fenics/shared/
        """)
    else:
        from _fem_common import _SymmetricFEM_Base

        class LanczosPotentialFEM(_SymmetricFEM_Base):
            def __init__(self, degree=1, mesh_name='eighth_of_sphere'):
                super(LanczosPotentialFEM, self).__init__(
                    degree=degree,
                    mesh_path=os.path.join(DIRNAME,
                                           'meshes',
                                           mesh_name))

            def _make_csd(self, degree, n):
                return Expression(f'''
                    x[0] >= n || x[1] >= n || x[2] >= n ?
                     0 :
                     a * (x[0] < {np.finfo(np.float32).eps} ? 1 : sin({np.pi} * x[0]) * sin({np.pi} * x[0] / n) / (x[0] * x[0] * {np.pi ** 2} / n))
                     * (x[1] < {np.finfo(np.float32).eps} ? 1 : sin({np.pi} * x[1]) * sin({np.pi} * x[1] / n) / (x[1] * x[1] * {np.pi ** 2} / n))
                     * (x[2] < {np.finfo(np.float32).eps} ? 1 : sin({np.pi} * x[2]) * sin({np.pi} * x[2] / n) / (x[2] * x[2] * {np.pi ** 2} / n))
                    ''',
                                 n=n,
                                 degree=degree,
                                 a=1.0)

            def potential_behind_dome(self, radius, n):
                return 0.25 / radius / np.pi

        logging.basicConfig(level=logging.INFO)

        if not os.path.exists(SOLUTION_DIRECTORY):
            os.makedirs(SOLUTION_DIRECTORY)

        mesh_name = sys.argv[1]

        fem = LanczosPotentialFEM(mesh_name=mesh_name)
        N = 1 + int(np.ceil(fem.RADIUS))

        for n in [1, 2, 3]:
            solution_filename = '{}_lanczos_{}.npz'.format(mesh_name,
                                                           n)
            stats = []
            results = {'N': N,
                       'folds': n,
                       'STATS': stats,
                       'radius': fem.RADIUS,
                       'sampling_frequency': SAMPLING_FREQUENCY,
                       }
            for degree in [1, 2, 3]:
                logger.info('Lanczos{} (deg={})'.format(n, degree))
                potential = fem(degree, n)

                stats.append((degree,
                              potential is not None,
                              fem.iterations,
                              fem.time.total_seconds()))
                logger.info('Lanczos{} (deg={}): {}'.format(n, degree,
                                                            'SUCCEED' if potential is not None else 'FAILED'))
                if potential is not None:
                    N_LIMIT = (N - 1) * SAMPLING_FREQUENCY + 1 # TODO: prove correctness
                    POTENTIAL = np.empty(N_LIMIT * (N_LIMIT + 1) * (N_LIMIT + 2) // 6)
                    POTENTIAL.fill(np.nan)
                    for x in range(N):
                        for y in range(x + 1):
                            for z in range(y + 1):
                                idx = x * (x + 1) * (x + 2) // 6 + y * (
                                            y + 1) // 2 + z
                                xx = x / float(SAMPLING_FREQUENCY)
                                yy = y / float(SAMPLING_FREQUENCY)
                                zz = z / float(SAMPLING_FREQUENCY)
                                r = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)
                                if r >= fem.RADIUS:
                                    v = fem.potential_behind_dome(r, n)
                                else:
                                    try:
                                        v = potential(xx, yy, zz)
                                    except RuntimeError as e:
                                        logger.warning("""
                                potential({}, {}, {})
                                (r = {})
                                raised:
                                {}""".format(xx, yy, zz, r, e))
                                        v = fem.potential_behind_dome(r, n)
                                POTENTIAL[idx] = v
                    results['Lanczos_{}'.format(degree)] = POTENTIAL
                    results['A_{}'.format(degree)] = fem.a
                    np.savez_compressed(LanczosSourceFactory.solution_path(solution_filename),
                                        **results)
