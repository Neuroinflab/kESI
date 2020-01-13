# $ docker run -ti --env HOST_UID=$(id -u) --env HOST_GID=$(id -g) -v $(pwd):/home/fenics/shared:Z quay.io/fenicsproject/stable
# fenics@...$ cd /home/fenics/shared/

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


SAMPLING_FREQUENCY = 64


class FiniteSliceGaussianSourceFactory(_fem_common._SourceFactory_Base):
    pass


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

            def __init__(self, degree=1, mesh_name='finite_slice'):
                super(GaussianPotentialFEM, self).__init__(
                      degree=degree,
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

            def get_linear_equation_matrix(self):
                del self._terms_with_unknown
                self._terms_with_unknown = assemble(self._a)
                return self._terms_with_unknown


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

                for degree in [1, 2]:  # 3 causes segmentation fault
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
                                   idx_x * (idx_x - 1) // 2 + idx_z] = fem.a
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
                                                          idx_x * (idx_x - 1) // 2 + idx_z,
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

                                np.savez_compressed(FiniteSliceGaussianSourceFactory.solution_path(
                                                        solution_filename,
                                                        False) + str(tmp_mark),
                                                    **results)
                                tmp_mark = 1 - tmp_mark

                np.savez_compressed(FiniteSliceGaussianSourceFactory.solution_path(
                                        solution_filename,
                                        False),
                                    **results)
