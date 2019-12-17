import datetime
import itertools
import logging
import warnings
import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator

DIRNAME = os.path.dirname(__file__)
SOLUTION_DIRECTORY = os.path.join(DIRNAME,
                                  'solutions')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

try:
    from dolfin import (Constant, Mesh, MeshFunction, FunctionSpace,
                        TestFunction, TrialFunction, Function,
                        Measure, inner, grad, assemble, KrylovSolver,
                        Expression, DirichletBC, XDMFFile, MeshValueCollection,
                        cpp)

except (ModuleNotFoundError, ImportError):
    logger.warning("Unable to import from dolfin")

else:
    class _SymmetricFEM_Base(object):
        RADIUS = 55.5
        EXTERNAL_SURFACE = 1
        MAX_ITER = 10000

        def __init__(self, degree=3, mesh_path=None):
            if mesh_path is not None:
                self.PATH = mesh_path

            logger.debug('Loading mesh...')

            # with XDMFFile(self.PATH + '.xdmf') as fh:
            #     self._mesh = Mesh()
            #     fh.read(self._mesh)
            #
            # with XDMFFile(self.PATH + '_boundaries.xdmf') as fh:
            #     mvc = MeshValueCollection("size_t", self._mesh, 2)
            #     fh.read(mvc, "boundaries")
            #     self._boundaries = cpp.mesh.MeshFunctionSizet(self._mesh, mvc)
            #
            # with XDMFFile(self.PATH + '_subdomains.xdmf') as fh:
            #     mvc = MeshValueCollection("size_t", self._mesh, 3)
            #     fh.read(mvc, "subdomains")
            #     self._subdomains = cpp.mesh.MeshFunctionSizet(self._mesh, mvc)

            self._mesh = Mesh(self.PATH + '.xml')
            self._subdomains = MeshFunction("size_t", self._mesh,
                                            self.PATH + '_physical_region.xml')
            self._boundaries = MeshFunction("size_t", self._mesh,
                                            self.PATH + '_facet_region.xml')

            logger.debug('Done.')
            self._change_degree(degree)

        def _change_degree(self, degree):
            logger.debug('Creating function space...')
            self._V = FunctionSpace(self._mesh, "CG", degree)
            logger.debug('Done.  Creating integration subdomains...')
            self._dx = Measure("dx")(subdomain_data=self._subdomains)
            logger.debug('Done.  Creating test function...')
            self._v = TestFunction(self._V)
            logger.debug('Done.  Creating potential function...')
            self._potential_function = Function(self._V)
            logger.debug('Done.  Creating trial function...')
            self._potential_trial = TrialFunction(self._V)
            logger.debug('Done.  Creating LHS part of equation...')
            self._a = inner(grad(self._potential_trial),
                            grad(self._v)) * self._dx
            logger.debug('Done.  Assembling linear equation matrix...')
            self._terms_with_unknown = assemble(self._a)
            self._degree = degree

            logger.debug('Done.  Creating solver...')
            self._solver = KrylovSolver("cg", "ilu")
            self._solver.parameters["maximum_iterations"] = self.MAX_ITER
            self._solver.parameters["absolute_tolerance"] = 1E-8
            logger.debug('Done.  Solver created.')

        def __call__(self, degree, *args, **kwargs):
            if degree != self._degree:
                self._change_degree(degree)

            logger.debug('Creating CSD expression...')
            csd = self._make_csd(degree, *args, **kwargs)
            logger.debug('Done.  Normalizing...')
            self.a = csd.a = (0.125 / assemble(csd
                                               * Measure("dx", self._mesh)))
            logger.debug('Done.  Creating RHS part of equation...')
            L = csd * self._v * self._dx
            logger.debug('Done.  Assembling linear equation vector...')
            known_terms = assemble(L)
            logger.debug('Done.  Defining boundary condition...')
            dirichlet_bc = DirichletBC(self._V,
                                       Constant(
                                                self.potential_behind_dome(
                                                         self.RADIUS,
                                                         *args, **kwargs)),
                                       self._boundaries,
                                       self.EXTERNAL_SURFACE)
            logger.debug('Done.  Copying linear equation matrix...')
            terms_with_unknown = self._terms_with_unknown.copy()
            logger.debug('Done.  Applying boundary condition...')
            dirichlet_bc.apply(terms_with_unknown, known_terms)
            logger.debug('Done.')

            start = datetime.datetime.now()
            try:
                logger.debug('Solving linear equation...')
                self.iterations = self._solver.solve(terms_with_unknown,
                                                    self._potential_function.vector(),
                                                    known_terms)
                logger.debug('Done.')
                return self._potential_function

            except RuntimeError as e:
                self.iterations = self.MAX_ITER
                logger.warning("Solver failed: {}".format(repr(e)))
                return None

            finally:
                self.time = datetime.datetime.now() - start


class _SymmetricSourceFactory_Base(object):
    def __init__(self, filename=None,
                 degree=1,
                 _limit=np.inf,
                 *args, **kwargs):
        """
        Parameters
        ----------
        filename : str
        degree
        _limit
        ground_truth : bool
        """
        with np.load(self.solution_path(filename)) as fh:
            self.load_specific_attributes(fh)
            COMPRESSED = fh[self.solution_array_name(degree, *args, **kwargs)]

            sampling_frequency = fh['sampling_frequency']

            N = min(_limit, fh['N'])

            self.a = fh['A_{}'.format(degree)]

            stride = 2 * N - 1
            linear_stride = (N - 1) * sampling_frequency + 1

            self.POTENTIAL = self.empty(stride ** 3)
            self.X = self.empty(stride ** 3)
            self.Y = self.empty(stride ** 3)
            self.Z = self.empty(stride ** 3)

            POTENTIAL = self.empty((linear_stride,
                                    linear_stride,
                                    linear_stride))

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

            for x in range(0, linear_stride):
                for y in range(0, x + 1):
                    for z in range(0, y + 1):
                        val = COMPRESSED[x * (x + 1) * (x + 2) // 6
                                         + y * (y + 1) // 2
                                         + z]
                        for i, j, k in itertools.permutations([x, y, z]):
                            POTENTIAL[i, j, k] = val

        self.interpolator = RegularGridInterpolator((np.linspace(0, (linear_stride - 1.0) / sampling_frequency, linear_stride),
                                                     np.linspace(0, (linear_stride - 1.0) / sampling_frequency, linear_stride),
                                                     np.linspace(0, (linear_stride - 1.0) / sampling_frequency, linear_stride)),
                                                    POTENTIAL)

    @classmethod
    def solution_path(cls, solution_filename):
        return os.path.join(SOLUTION_DIRECTORY,
                            solution_filename)

    def empty(self, shape):
        X = np.empty(shape)
        X.fill(np.nan)
        return X

    def potential_sinc(self, X, Y, Z):
        return np.inner((np.sinc(np.subtract.outer(X, self.X))
                         * np.sinc(np.subtract.outer(Y, self.Y))
                         * np.sinc(np.subtract.outer(Z, self.Z))),
                        self.POTENTIAL)

    # TODO: handle cases of distant points
    def potential_linear(self, X, Y, Z):
        return self.interpolator(np.stack((abs(X),
                                           abs(Y),
                                           abs(Z)),
                                          axis=-1))

    def Source(self, *args, **kwargs):
        warnings.warn('The factory is a callable.  Call it instead.',
                      DeprecationWarning)
        return self(*args, **kwargs)

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
            return (self._pre_normalize(self._parent.csd, X, Y, Z)
                    / self._scale ** 3)

        def _pre_normalize(self, f, X, Y, Z):
            return f((X - self._x) / self._scale,
                     (Y - self._y) / self._scale,
                     (Z - self._z) / self._scale)

        def _normalize_potential(self, f, X, Y, Z):
            return (self._pre_normalize(f, X, Y, Z)
                    / (self._scale * self._conductivity))

        def potential_sinc(self, X, Y, Z):
            return self._normalize_potential(self._parent.potential_sinc,
                                             X, Y, Z)

        def potential(self, X, Y, Z):
            return self.potential_linear(X, Y, Z)

        def potential_linear(self, X, Y, Z):
            return self._normalize_potential(self._parent.potential_linear,
                                             X, Y, Z)