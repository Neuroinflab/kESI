# $ docker run -ti --env HOST_UID=$(id -u) --env HOST_GID=$(id -g) -v $(pwd):/home/fenics/shared:Z quay.io/fenicsproject/stable
# fenics@...$ cd /home/fenics/shared/

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


class FiniteSliceGaussianSourceFactory(_fem_common._SourceFactory_Base):
    def __init__(self, filename=None,
                 degree=1,
                 try_local_first=True):
         with np.load(self.solution_path(filename,
                                         try_local_first)) as fh:
             self.slice_thickness = fh['slice_thickness']
             k = fh['k']
             self.standard_deviation = self.slice_thickness / 2 ** k
             self.X = list(np.linspace(self.standard_deviation / 2,
                                       self.slice_thickness - self.standard_deviation / 2,
                                       2**k))
             self.sampling_frequency = fh['sampling_frequency']
             self.a = fh['A_{}'.format(degree)]
             self.POTENTIAL = fh[self.solution_array_name(degree)]

             self.X_SAMPLE = np.linspace(-self.slice_thickness,
                                         self.slice_thickness,
                                         2 * self.sampling_frequency + 1)
             self.Y_SAMPLE = np.linspace(0,
                                         self.slice_thickness,
                                         self.sampling_frequency + 1)

         self.slice_radius = 3.0e-3  # m

    def solution_array_name(self, degree):
        return 'Gaussian_{}'.format(degree)

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

        return self._Source(x, y, z, self.a[idx_y, idx], POTENTIAL, self)

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
        $ docker run -ti --env HOST_UID=$(id -u) --env HOST_GID=$(id -g) -v $(pwd):/home/fenics/shared:Z quay.io/fenicsproject/stable
        $ cd /home/fenics/shared/
        """)
    else:
        class GaussianPotentialFEM(_fem_common._FEM_Base):
            _RADIUS = {'finite_slice': 0.3,
                       'finite_slice_small': 0.03,
                       'finite_slice_smaller': 0.003,
                       }  # m
            SLICE_RADIUS = 3.0e-3  # m
            SLICE_THICKNESS = 0.3e-3  # m
            SLICE_VOLUME = 1
            SALINE_VOLUME = 2
            EXTERNAL_SURFACE = 3
            CONDUCTIVITY = {SLICE_VOLUME: 0.3,  # S / m
                            SALINE_VOLUME: 1.5,  # S / m
                            }

            def __init__(self, mesh_name='finite_slice'):
                super(GaussianPotentialFEM, self).__init__(
                      mesh_path=os.path.join(_fem_common.DIRNAME,
                                             'meshes',
                                             mesh_name))
                self.RADIUS = self._RADIUS[mesh_name]

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
                                   x[1] > {self.SLICE_THICKNESS}
                                   || x[0] * x[0] + x[2] * x[2] > {self.SLICE_RADIUS ** 2}
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


        logging.basicConfig(level=logging.INFO)

        if not os.path.exists(_fem_common.SOLUTION_DIRECTORY):
            os.makedirs(_fem_common.SOLUTION_DIRECTORY)

        for mesh_name in sys.argv[1:]:
            fem = GaussianPotentialFEM(mesh_name=mesh_name)

            K_MAX = 3  # as element size is SLICE_THICKNESS / 32,
                       # the smallest sd considered safe is
                       # SLICE_THICKNESS / (2 ** 3)
            for k in range(K_MAX + 1):
                sd = fem.SLICE_THICKNESS / 2 ** k
                solution_filename = '{}_gaussian_{:04d}.npz'.format(mesh_name,
                                                                    int(round(1000 / 2 ** k)))
                tmp_mark = 0
                X = np.linspace(sd / 2, fem.SLICE_THICKNESS - sd / 2, 2**k)
                stats = []
                results = {'k': k,
                           'slice_thickness': fem.SLICE_THICKNESS,
                           'slice_radius': fem.SLICE_RADIUS,
                           'STATS': stats,
                           'sampling_frequency': SAMPLING_FREQUENCY,
                           }

                for degree in [1, 2]:  # 3 causes segmentation fault or takes 40 mins
                    logger.info('Gaussian SD={} (deg={})'.format(sd, degree))
                    POTENTIAL = np.empty((2**k,
                                          2**k * (2 ** k + 1) // 2,
                                          2 * SAMPLING_FREQUENCY + 1,
                                          SAMPLING_FREQUENCY + 1,
                                          2 * SAMPLING_FREQUENCY + 1))
                    POTENTIAL.fill(np.nan)
                    results['Gaussian_{}'.format(degree)] = POTENTIAL
                    AS = np.empty((2**k, 2**k * (2 ** k + 1) // 2))
                    AS.fill(np.nan)
                    results['A_{}'.format(degree)] = AS

                    save_stopwatch = _fem_common.Stopwatch()

                    with _fem_common.Stopwatch() as unsaved_time:
                        for idx_y, src_y in enumerate(X):
                            for idx_x, src_x in enumerate(X):
                                for idx_z, src_z in enumerate(X[:idx_x+1]):
                                    logger.info(
                                        'Gaussian SD={}, x={}, y={}, z={} (deg={})'.format(
                                            sd,
                                            src_x,
                                            src_y,
                                            src_z,
                                            degree))
                                    potential = fem(degree, src_x, src_y, src_z, sd)

                                    stats.append((degree,
                                                  src_x,
                                                  src_y,
                                                  src_z,
                                                  potential is not None,
                                                  fem.iterations,
                                                  fem.time.total_seconds()))

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

                                    AS[idx_y,
                                       idx_x * (idx_x + 1) // 2 + idx_z] = fem.a
                                    if potential is not None:
                                        for i, x in enumerate(np.linspace(-fem.SLICE_THICKNESS,
                                                                fem.SLICE_THICKNESS,
                                                                2 * SAMPLING_FREQUENCY + 1)):
                                            for j, y in enumerate(np.linspace(0,
                                                                    fem.SLICE_THICKNESS,
                                                                    SAMPLING_FREQUENCY + 1)):
                                                for kk, z in enumerate(np.linspace(-fem.SLICE_THICKNESS,
                                                                        fem.SLICE_THICKNESS,
                                                                        2 * SAMPLING_FREQUENCY + 1)):
                                                    POTENTIAL[idx_y,
                                                              idx_x * (idx_x + 1) // 2 + idx_z,
                                                              i,
                                                              j,
                                                              kk] = potential(x, y, z)
                                    logger.info('Gaussian SD={}, x={}, y={}, z={} (deg={}): {}\t({fem.iterations}, {fem.time})'.format(
                                                sd,
                                                src_x,
                                                src_y,
                                                src_z,
                                                degree,
                                                'SUCCEED' if potential is not None else 'FAILED',
                                                fem=fem))
                                    if float(unsaved_time) > 10 * float(save_stopwatch):
                                        with save_stopwatch:
                                            np.savez_compressed(FiniteSliceGaussianSourceFactory.solution_path(
                                                                    solution_filename,
                                                                    False) + str(tmp_mark),
                                                                **results)
                                        unsaved_time.reset()
                                        tmp_mark = 1 - tmp_mark

                np.savez_compressed(FiniteSliceGaussianSourceFactory.solution_path(
                                        solution_filename,
                                        False),
                                    **results)
