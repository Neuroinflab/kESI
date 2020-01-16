import datetime
import itertools
import logging
import warnings
import os
import gc

import numpy as np
from scipy.interpolate import RegularGridInterpolator

DIRNAME = os.path.dirname(__file__)
SOLUTION_DIRECTORY = os.path.join(DIRNAME,
                                  'solutions')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Stopwatch(object):
    def __init__(self):
        self.reset()
        self._running = False

    def reset(self):
        self.start = datetime.datetime.now()
        self._end = self.start

    @property
    def end(self):
        if self.running:
            return datetime.datetime.now()

        return self._end

    @property
    def duration(self):
        return self.end - self.start

    def __float__(self):
        return self.duration.total_seconds()

    def __enter__(self):
        self.reset()
        self._running = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end = datetime.datetime.now()
        self._running = False


try:
    from dolfin import (Constant, Mesh, MeshFunction, FunctionSpace,
                        TestFunction, TrialFunction, Function,
                        Measure, inner, grad, assemble, KrylovSolver,
                        Expression, DirichletBC, XDMFFile, MeshValueCollection,
                        cpp)

except (ModuleNotFoundError, ImportError):
    logger.warning("Unable to import from dolfin")

else:
    class _FEM_Base(object):
        MAX_ITER = 10000

        def __init__(self, mesh_path=None):
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
            self._degree = None

        def _change_degree(self, degree, *args, **kwargs):
            gc.collect()
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
            self._a = self._lhs()
            logger.debug('Done.  Assembling linear equation matrix...')
            self._terms_with_unknown = assemble(self._a)
            logger.debug('Done.  Defining boundary condition...')
            self._dirichlet_bc = self._boundary_condition(*args, **kwargs)
            logger.debug('Done.  Applying boundary condition to the matrix...')
            self._dirichlet_bc.apply(self._terms_with_unknown)

            logger.debug('Done.  Creating solver...')
            self._solver = KrylovSolver("cg", "ilu")
            self._solver.parameters["maximum_iterations"] = self.MAX_ITER
            self._solver.parameters["absolute_tolerance"] = 1E-8
            logger.debug('Done.  Solver created.')

            self._degree = degree

        def __call__(self, degree, *args, **kwargs):
            if degree != self._degree:
                self._change_degree(degree, *args, **kwargs)

            gc.collect()
            logger.debug('Creating CSD expression...')
            csd = self._make_csd(degree, *args, **kwargs)
            logger.debug('Done.  Normalizing...')
            self.a = csd.a = self._csd_normalization_factor(csd)
            logger.debug('Done.  Creating RHS part of equation...')
            L = csd * self._v * self._dx
            logger.debug('Done.  Assembling linear equation vector...')
            known_terms = assemble(L)
            logger.debug('Done.  Applying boundary condition to the vector...')
            self._dirichlet_bc.apply(known_terms)
            logger.debug('Done.')

            start = datetime.datetime.now()
            try:
                logger.debug('Solving linear equation...')
                gc.collect()
                self.iterations = self._solver.solve(self._terms_with_unknown,
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


    class _SymmetricFEM_Base(_FEM_Base):
        RADIUS = 55.5
        EXTERNAL_SURFACE = 1

        def _lhs(self):
            return inner(grad(self._potential_trial),
                         grad(self._v)) * self._dx

        def _csd_normalization_factor(self, csd):
            old_a = csd.a
            csd.a = 1
            try:
                return (0.125 / assemble(csd
                                         * Measure("dx", self._mesh)))
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


class _SourceFactory_Base(object):
    @classmethod
    def solution_path(cls, solution_filename, try_local_first=True):
        if try_local_first and os.path.exists(solution_filename):
            return solution_filename

        return os.path.join(SOLUTION_DIRECTORY,
                            solution_filename)


class _SymmetricSourceFactory_Base(_SourceFactory_Base):
    def __init__(self, filename=None,
                 degree=1,
                 try_local_first=True,
                 approximation_mixing_threshold=0.5,
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
        self._appoximation_mixing_threshold = approximation_mixing_threshold
        with np.load(self.solution_path(filename,
                                        try_local_first)) as fh:
            self.load_specific_attributes(fh)
            self.a = fh['A_{}'.format(degree)]

            COMPRESSED = fh[self.solution_array_name(degree, *args, **kwargs)]
            sampling_frequency = fh['sampling_frequency']

            N = min(_limit, fh['N'])

            stride = 2 * N - 1
            linear_stride = (N - 1) * sampling_frequency + 1

            self.POTENTIAL = self.empty(stride ** 3)
            self.X = self.empty(stride ** 3)
            self.Y = self.empty(stride ** 3)
            self.Z = self.empty(stride ** 3)

            POTENTIAL = self.empty((linear_stride,
                                    linear_stride,
                                    linear_stride))

            # WARNING: subsampling without filtering
            for x in range(0, N * sampling_frequency, sampling_frequency):
                for y in range(0, x + 1, sampling_frequency):
                    for z in range(0, y + 1, sampling_frequency):
                        val = COMPRESSED[x * (x + 1) * (x + 2) // 6
                                         + y * (y + 1) // 2
                                         + z]
                        # NOTE: if x == y, x == z or y == z
                        # itertools.permutations() may repeat xs, ys, zs
                        for xs, ys, zs in itertools.permutations(
                                [self._abs_inv_list(x // sampling_frequency),
                                 self._abs_inv_list(y // sampling_frequency),
                                 self._abs_inv_list(z // sampling_frequency),
                                 ]):
                            for i, j, k in itertools.product(xs, ys, zs):
                                idx = (((N - 1 + i)
                                        * stride + N - 1 + j)
                                       * stride) + N - 1 + k

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

        self._radius = (linear_stride - 1.0) / sampling_frequency
        LINSPACE = np.linspace(0, self._radius, linear_stride)
        self._interpolator = RegularGridInterpolator((LINSPACE,
                                                      LINSPACE,
                                                      LINSPACE),
                                                     POTENTIAL,
                                                     bounds_error=False)

    def _abs_inv_list(self, x):
        if x == 0:
            return [x]
        return [-x, x]

    def empty(self, shape):
        X = np.empty(shape)
        X.fill(np.nan)
        return X

    def potential_sinc(self, X, Y, Z):
        # self.POTENTIAL is already downsampled thus there is no need
        # of accounting for sampling frequency
        return np.inner((np.sinc(np.subtract.outer(X, self.X))
                         * np.sinc(np.subtract.outer(Y, self.Y))
                         * np.sinc(np.subtract.outer(Z, self.Z))),
                        self.POTENTIAL)

    def potential_sinc_scalar(self, x, y, z):
        w = max(abs(w) for w in [x, y, z])
        if w < self._radius:
            SINC_3D = (np.sinc(x - self.X)
                       * np.sinc(y - self.Y)
                       * np.sinc(z - self.Z))
            # self.POTENTIAL is already downsampled thus there is no need
            # of accounting for sampling frequency
            interpolated = (SINC_3D * self.POTENTIAL).sum()

            if w < self._appoximation_mixing_threshold * self._radius:
                return interpolated

        approximated = self.potential_behind_dome(np.sqrt(x**2 + y**2 + z**2))
        if w >= self._radius:
            return approximated

        q = ((self._radius - w)
             / ((1.0 - self._appoximation_mixing_threshold) * self._radius))
        return q * interpolated + (1 - q) * approximated

    def potential_behind_dome(self, r):
        return 0.25 / np.pi / r

    def potential_linear(self, X, Y, Z):
        abs_X = abs(X)
        abs_Y = abs(Y)
        abs_Z = abs(Z)
        W = np.maximum(abs_X, np.maximum(abs_Y, abs_Z))
        Q = (self._radius - W) / ((1.0 - self._appoximation_mixing_threshold) * self._radius)
        INTERPOLATED = self._interpolator(np.stack((abs_X, abs_Y, abs_Z),
                                                   axis=-1))
        APPROXIMATED = self.potential_behind_dome(np.sqrt(np.square(abs_X)
                                                          + np.square(abs_Y)
                                                          + np.square(abs_Z)))
        return np.where(W <= self._radius * self._appoximation_mixing_threshold,
                        INTERPOLATED,
                        np.where(W <= self._radius,
                                 Q * INTERPOLATED + (1.0 - Q) * APPROXIMATED,
                                 APPROXIMATED))

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

        def potential_sinc_scalar(self, X, Y, Z):
            return self._normalize_potential(self._parent.potential_sinc_scalar,
                                             X, Y, Z)

        def potential(self, X, Y, Z):
            return self.potential_linear(X, Y, Z)

        def potential_linear(self, X, Y, Z):
            return self._normalize_potential(self._parent.potential_linear,
                                             X, Y, Z)