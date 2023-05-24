#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
#    Copyright (C) 2019-2023 Jakub M. Dzik (Laboratory of Neuroinformatics;   #
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
# import itertools

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import numpy as np

try:
    from extras.local.fem import _common as fc
    # When run as script raises:
    #  - `ImportError` (Python 3.6-9), or
    #  - `SystemError` (Python 3.3-5), or
    #  - `ValueError` (Python 2.7).

except ImportError:
    import extras.local.fem._common as fc

SOLUTION_FILENAME = 'Slice.npz'
MAX_ITER = 10000


# class GaussianSourceSliceFactory(object):
#     def __init__(self, degree=3, _limit=np.inf):
#         with np.load(SOLUTION_FILENAME) as fh:
#             POTENTIAL = fh['Gaussian_{}'.format(degree)]
#             N = min(_limit, fh['N'])
#
#             self.a = fh['A_{}_{}'.format(n, degree)]
#             COMPRESSED = fh['Lanczos{}_{}'.format(n, degree)]
#
#             stride = 2 * N - 1
#             # POTENTIAL = fc.empty_array([stride, stride, stride])
#             self.POTENTIAL = fc.empty_array(stride ** 3)
#             self.X = fc.empty_array(stride ** 3)
#             self.Y = fc.empty_array(stride ** 3)
#             self.Z = fc.empty_array(stride ** 3)
#
#             for x in range(N):
#                 for y in range(x + 1):
#                     for z in range(y + 1):
#                         val = COMPRESSED[x * (x + 1) * (x + 2) // 6
#                                          + y * (y + 1) // 2
#                                          + z]
#
#                         for xs, ys, zs in itertools.permutations([[x] if x == 0 else [-x, x],
#                                                                   [y] if y == 0 else [-y, y],
#                                                                   [z] if z == 0 else [-z, z]]):
#                             # if x == y, x == z or y == z may repeat xs, ys, zs
#                             for i, j, k in itertools.product(xs, ys, zs):
#                                 idx = ((N - 1 + i) * stride + N - 1 + j) * stride + N - 1 + k
#
#                                 self.POTENTIAL[idx] = val
#                                 self.X[idx] = i
#                                 self.Y[idx] = j
#                                 self.Z[idx] = k
#
#                                     # POTENTIAL[N - 1 + i,
#                                     #           N - 1 + j,
#                                     #           N - 1 + k] = val
#         # X, Y, Z = np.meshgrid(np.arange(1 - N, 2 * N),
#         #                       np.arange(1 - N, 2 * N),
#         #                       np.arange(1 - N, 2 * N),
#         #                       indexing='ij')
#         # self.POTENTIAL = POTENTIAL.flatten()
#         # self.X = X.flatten()
#         # self.Y = Y.flatten()
#         # self.Z = Z.flatten()
#
#     def _lanczos(self, X):
#         return np.where(abs(X) >= self.n,
#                         0,
#                         np.sinc(X) * np.sinc(X / self.n))
#
#     def csd(self, X, Y, Z):
#         return self._lanczos(X) * self._lanczos(Y) * self._lanczos(Z) * self.a
#
#     def potential(self, X, Y, Z):
#         return np.inner((np.sinc(np.subtract.outer(X, self.X))
#                          * np.sinc(np.subtract.outer(Y, self.Y))
#                          * np.sinc(np.subtract.outer(Z, self.Z))),
#                         self.POTENTIAL)
#
#     class _Source(object):
#         def __init__(self,
#                      scale,
#                      conductivity,
#                      x,
#                      y,
#                      z,
#                      parent):
#             self._scale = scale
#             self._conductivity = conductivity
#             self._x = x
#             self._y = y
#             self._z = z
#             self._parent = parent
#
#         def csd(self, X, Y, Z):
#             return (self._parent.csd((X - self._x) / self._scale,
#                                      (Y - self._y) / self._scale,
#                                      (Z - self._z) / self._scale)
#                     / self._scale ** 3)
#
#         def potential(self, X, Y, Z):
#             return (self._parent.potential((X - self._x) / self._scale,
#                                            (Y - self._y) / self._scale,
#                                            (Z - self._z) / self._scale)
#                     / (self._scale * self._conductivity))
#
#     def Source(self, x=0, y=0, z=0, scale=1, conductivity=1):
#         return self._Source(scale, conductivity,
#                             x, y, z,
#                             self)



if __name__ == '__main__':
    import datetime
    try:
        from dolfin import Constant, Mesh, MeshFunction, FunctionSpace, TestFunction, TrialFunction, Function, Measure, inner, grad, assemble, KrylovSolver
        from dolfin import Expression, DirichletBC

    except (ModuleNotFoundError, ImportError):
        print("""Run docker first:
        $ docker run -ti --env HOST_UID=$(id -u) --env HOST_GID=$(id -g) \\
                     -v $(pwd):/home/fenics/shared:Z \\
                     -v $(pwd)/solutions:/home/fenics/shared/solutions:Z \\
                     -w /home/fenics/shared \\
                     quay.io/fenicsproject/stable
        """)
    else:
        class GaussianPotentialSliceFEM(object):
            PATH = 'meshes/slice'
            H = 3e-4  # m
            RADIUS = 1.0  # m
            SLICE_CONDUCTIVITY = 0.3  # S / m
            SALINE_CONDUCTIVITY = 1.5  # S / m

            SALINE_VOL = 3
            SLICE_VOL = 4
            ROI_VOL = 5

            MODEL_DOME = 2
            MODEL_BASE = 1

            def __init__(self):
                self._mesh = Mesh(self.PATH + '.xml')
                self._subdomains = MeshFunction("size_t", self._mesh,
                                                self.PATH + '_physical_region.xml')
                self._boundaries = MeshFunction("size_t", self._mesh,
                                                self.PATH + '_facet_region.xml')

            def __call__(self, degree=3, y=0., standard_deviation=1.):
                V = FunctionSpace(self._mesh, "CG", degree)
                v = TestFunction(V)
                potential_trial = TrialFunction(V)
                potential = Function(V)
                dx = Measure("dx")(subdomain_data=self._subdomains)
                # ds = Measure("ds")(subdomain_data=self._boundaries)
                csd = Expression(f'''
                                  x[1] >= {self.H} ?
                                  0 :
                                  a * exp({-0.5 / standard_deviation ** 2}
                                          * ((x[0])*(x[0])
                                             + (x[1] - {y})*(x[1] - {y})
                                             + (x[2])*(x[2])
                                             ))
                                  ''',
                                 degree=degree,
                                 a=1.0)

                self.a = csd.a = 1.0 / assemble(csd * Measure("dx", self._mesh))
                # print(assemble(csd * Measure("dx", self._mesh)))
                L = csd * v * dx

                known_terms = assemble(L)
                # a = (inner(grad(potential_trial), grad(v))
                #      * (Constant(self.SALINE_CONDUCTIVITY) * dx(self.SALINE_VOL)
                #         + Constant(self.SLICE_CONDUCTIVITY) * (dx(self.SLICE_VOL)
                #                                                + dx(self.ROI_VOL))))
                a = sum(Constant(conductivity)
                        * inner(grad(potential_trial), grad(v))
                        * dx(domain)
                        for domain, conductivity
                        in [(self.SALINE_VOL, self.SALINE_CONDUCTIVITY),
                            (self.SLICE_VOL, self.SLICE_CONDUCTIVITY),
                            (self.ROI_VOL, self.SLICE_CONDUCTIVITY),
                            ])
                terms_with_unknown = assemble(a)
                dirchlet_bc = DirichletBC(V,
                                          Constant(2.0 * 0.25
                                                   / (
                                                               self.RADIUS * np.pi * self.SALINE_CONDUCTIVITY)),
                                          # 2.0 becaue of dielectric base duplicating
                                          # the current source
                                          # slice conductivity and thickness considered
                                          # negligible
                                          self._boundaries,
                                          self.MODEL_DOME)
                dirchlet_bc.apply(terms_with_unknown, known_terms)
                solver = KrylovSolver("cg", "ilu")
                solver.parameters["maximum_iterations"] = MAX_ITER
                solver.parameters["absolute_tolerance"] = 1E-8
                # solver.parameters["monitor_convergence"] = True
                start = datetime.datetime.now()
                try:
                    self.iterations = solver.solve(terms_with_unknown,
                                                   potential.vector(),
                                                   known_terms)
                    return potential

                except RuntimeError as e:
                    self.iterations = MAX_ITER
                    logger.warning("Solver failed: {}".format(repr(e)))
                    return None

                finally:
                    self.time = datetime.datetime.now() - start


        logging.basicConfig(level=logging.INFO)

        fem = GaussianPotentialSliceFEM()

        SD = fem.H * np.logspace(-5, -3, 3, base=2)
        H = np.linspace(0, fem.H, 33)
        X = np.linspace(0, fem.H, 33)
        Y = np.linspace(0, fem.H, 33)

        stats = []
        results = {'SD': SD,
                   'H': H,
                   'X': X,
                   'Y': Y,
                   'STATS': stats,
                   }
        for degree in [1, 2, 3]:
            results['A_{}'.format(degree)] = []

            RES = fc.empty_array((len(SD),
                                  len(H),
                                  len(X),
                                  len(Y)))
            results['Gaussian_{}'.format(degree)] = RES

            for i_sd, sd in enumerate(SD):
                for i_h, h in enumerate(H):
                    logger.info('Gaussian (deg={}, sd={}, h={})'.format(degree,
                                                                        sd,
                                                                        h))
                    potential = fem(degree, y=h, standard_deviation=sd)

                    stats.append((degree,
                                  sd,
                                  h,
                                  potential is not None,
                                  fem.iterations,
                                  fem.time.total_seconds()))
                    logger.info('Gaussian (deg={}, sd={}, h={}): {}'.format(
                        degree,
                        sd,
                        h,
                        'SUCCEED' if potential is not None else 'FAILED'))

                    if potential is not None:
                        for i_x, x in enumerate(X):
                            for i_y, y in enumerate(Y):
                                RES[i_sd, i_h, i_x, i_y] = potential(x, y, 0)

                    results['A_{}'.format(degree)].append(fem.a)
                    np.savez_compressed(SOLUTION_FILENAME, **results)
