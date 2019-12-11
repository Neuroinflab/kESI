import datetime
import logging


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
