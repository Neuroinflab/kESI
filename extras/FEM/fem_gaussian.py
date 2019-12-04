# $ docker run -ti --env HOST_UID=$(id -u) --env HOST_GID=$(id -g) -v $(pwd):/home/fenics/shared:Z quay.io/fenicsproject/stable
# fenics@...$ cd /home/fenics/shared/

import os
import logging
import itertools

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import numpy as np

SOLUTION_FILENAME = os.path.join(os.path.dirname(__file__),
                                 'Gaussian.npz')
MAX_ITER = 10000


class GaussianSourceFactory(object):
    def __init__(self, standard_deviation=1., degree=3, _limit=np.inf):
        self.standard_deviation = standard_deviation

        with np.load(SOLUTION_FILENAME) as fh:
            N = min(_limit, fh['N'])

            self.a = fh['A_{:01.3f}_{}'.format(standard_deviation, degree)]
            COMPRESSED = fh['Gaussian_SD_{:01.3f}_{}'.format(standard_deviation,
                                                             degree)]

            stride = 2 * N - 1
            # POTENTIAL = self.empty([stride, stride, stride])
            self.POTENTIAL = self.empty(stride ** 3)
            self.X = self.empty(stride ** 3)
            self.Y = self.empty(stride ** 3)
            self.Z = self.empty(stride ** 3)

            for x in range(N):
                for y in range(x + 1):
                    for z in range(y + 1):
                        val = COMPRESSED[x * (x + 1) * (x + 2) // 6
                                         + y * (y + 1) // 2
                                         + z]

                        for xs, ys, zs in itertools.permutations([[x] if x == 0 else [-x, x],
                                                                  [y] if y == 0 else [-y, y],
                                                                  [z] if z == 0 else [-z, z]]):
                            # if x == y, x == z or y == z may repeat xs, ys, zs
                            for i, j, k in itertools.product(xs, ys, zs):
                                idx = ((N - 1 + i) * stride + N - 1 + j) * stride + N - 1 + k

                                self.POTENTIAL[idx] = val
                                self.X[idx] = i
                                self.Y[idx] = j
                                self.Z[idx] = k

                                    # POTENTIAL[N - 1 + i,
                                    #           N - 1 + j,
                                    #           N - 1 + k] = val
        # X, Y, Z = np.meshgrid(np.arange(1 - N, 2 * N),
        #                       np.arange(1 - N, 2 * N),
        #                       np.arange(1 - N, 2 * N),
        #                       indexing='ij')
        # self.POTENTIAL = POTENTIAL.flatten()
        # self.X = X.flatten()
        # self.Y = Y.flatten()
        # self.Z = Z.flatten()

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

    def potential(self, X, Y, Z):
        return np.inner((np.sinc(np.subtract.outer(X, self.X))
                         * np.sinc(np.subtract.outer(Y, self.Y))
                         * np.sinc(np.subtract.outer(Z, self.Z))),
                        self.POTENTIAL)

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

        def potential(self, X, Y, Z):
            return (self._parent.potential((X - self._x) / self._scale,
                                           (Y - self._y) / self._scale,
                                           (Z - self._z) / self._scale)
                    / (self._scale * self._conductivity))

    def Source(self, x=0, y=0, z=0, standard_deviation=1, conductivity=1):
        return self._Source(standard_deviation / self.standard_deviation,
                            conductivity,
                            x, y, z,
                            self)



if __name__ == '__main__':
    import datetime
    from scipy.special import erf

    try:
        from dolfin import Constant, Mesh, MeshFunction, FunctionSpace, TestFunction, TrialFunction, Function, Measure, inner, grad, assemble, KrylovSolver
        from dolfin import Expression, DirichletBC

    except (ModuleNotFoundError, ImportError):
        print("""Run docker first:
        $ docker run -ti --env HOST_UID=$(id -u) --env HOST_GID=$(id -g) -v $(pwd):/home/fenics/shared:Z quay.io/fenicsproject/stable
        $ cd /home/fenics/shared/
        """)
    else:
        class GaussianPotentialFEM(object):
            PATH = 'meshes/eighth_of_sphere'
            EXTERNAL_SURFACE = 1
            RADIUS = 55.5

            def __init__(self):
                self._mesh = Mesh(self.PATH + '.xml')
                self._subdomains = MeshFunction("size_t", self._mesh,
                                                self.PATH + '_physical_region.xml')
                self._boundaries = MeshFunction("size_t", self._mesh,
                                                self.PATH + '_facet_region.xml')

            def __call__(self,
                         standard_deviation=1.,
                         degree=3):
                V = FunctionSpace(self._mesh, "CG", degree)
                v = TestFunction(V)
                potential_trial = TrialFunction(V)
                potential = Function(V)
                dx = Measure("dx")(subdomain_data=self._subdomains)
                # ds = Measure("ds")(subdomain_data=self._boundaries)
                csd = Expression(f'''
                                 a * exp({-0.5 / standard_deviation ** 2}
                                         * ((x[0])*(x[0])
                                            + (x[1])*(x[1])
                                            + (x[2])*(x[2])
                                            ))
                                 ''',
                                 degree=degree,
                                 a=1.0)
                self.a = csd.a = (0.125
                                  / assemble(csd * Measure("dx", self._mesh)))
                # print(assemble(csd * Measure("dx", self._mesh)))
                L = csd * v * dx
                known_terms = assemble(L)
                a = inner(grad(potential_trial), grad(v)) * dx
                terms_with_unknown = assemble(a)
                dirchlet_bc = DirichletBC(V,
                                          Constant(0.25
                                                   * erf(self.RADIUS
                                                         / (np.sqrt(2)
                                                            * standard_deviation))
                                                   / (self.RADIUS * np.pi)),
                                          self._boundaries,
                                          self.EXTERNAL_SURFACE)
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

        fem = GaussianPotentialFEM()
        N = 1 + int(np.floor(fem.RADIUS / np.sqrt(3)))
        stats = []
        results = {'N': N,
                   'STATS': stats,
                   }
        for degree in [1, 2, 3]:
            for sd in [0.25, 0.5, 1, 2]:
                logger.info('Gaussian SD={} (deg={})'.format(sd, degree))
                potential = fem(sd, degree)

                stats.append((sd,
                              degree,
                              potential is not None,
                              fem.iterations,
                              fem.time.total_seconds()))
                logger.info('Gaussian SD={} (deg={}): {}'.format(sd, degree,
                                                            'SUCCEED' if potential is not None else 'FAILED'))
                if potential is not None:
                    POTENTIAL = np.empty(N * (N + 1) * (N + 2) // 6)
                    POTENTIAL.fill(np.nan)
                    for x in range(N):
                        for y in range(x + 1):
                            for z in range(y + 1):
                                POTENTIAL[x * (x + 1) * (x + 2) // 6
                                          + y * (y + 1) // 2
                                          + z] = potential(x, y, z)
                    results['Gaussian_SD_{:01.3f}_{}'.format(sd, degree)] = POTENTIAL
                    results['A_{:01.3f}_{}'.format(sd, degree)] = fem.a
                    np.savez_compressed(SOLUTION_FILENAME, **results)
