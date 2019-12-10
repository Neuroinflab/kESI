import datetime
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from dolfin import (Constant, Mesh, MeshFunction, FunctionSpace,
                        TestFunction, TrialFunction, Function,
                        Measure, inner, grad, assemble, KrylovSolver,
                        Expression, DirichletBC)

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

            self._mesh = Mesh(self.PATH + '.xml')
            self._subdomains = MeshFunction("size_t", self._mesh,
                                            self.PATH + '_physical_region.xml')
            self._boundaries = MeshFunction("size_t", self._mesh,
                                            self.PATH + '_facet_region.xml')

            self._change_degree(degree)

        def _change_degree(self, degree):
            self._V = FunctionSpace(self._mesh, "CG", degree)

            self._dx = Measure("dx")(subdomain_data=self._subdomains)
            self._v = TestFunction(self._V)
            self._potential_function = Function(self._V)
            self._potential_trial = TrialFunction(self._V)
            self._a = inner(grad(self._potential_trial),
                            grad(self._v)) * self._dx
            self._terms_with_unknown = assemble(self._a)
            self._degree = degree

            self._solver = KrylovSolver("cg", "ilu")
            self._solver.parameters["maximum_iterations"] = self.MAX_ITER
            self._solver.parameters["absolute_tolerance"] = 1E-8

        def __call__(self, degree, *args, **kwargs):
            if degree != self._degree:
                self._change_degree(degree)

            csd = self._make_csd(degree, *args, **kwargs)
            self.a = csd.a = (0.125 / assemble(csd
                                               * Measure("dx", self._mesh)))
            L = csd * self._v * self._dx
            known_terms = assemble(L)

            dirichlet_bc = DirichletBC(self._V,
                                       Constant(
                                                self._potential_behind_dome(
                                                         self.RADIUS,
                                                         *args, **kwargs)),
                                       self._boundaries,
                                       self.EXTERNAL_SURFACE)

            terms_with_unknown = self._terms_with_unknown.copy()
            dirichlet_bc.apply(terms_with_unknown, known_terms)

            start = datetime.datetime.now()
            try:
                self.iterations = self._solver.solve(terms_with_unknown,
                                                    self._potential_function.vector(),
                                                    known_terms)
                return self._potential_function

            except RuntimeError as e:
                self.iterations = self.MAX_ITER
                logger.warning("Solver failed: {}".format(repr(e)))
                return None

            finally:
                self.time = datetime.datetime.now() - start
