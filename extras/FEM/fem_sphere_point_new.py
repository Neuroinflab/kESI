#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
#    Copyright (C) 2019-2020 Jakub M. Dzik (Laboratory of Neuroinformatics;   #
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
import os

import numpy as np

try:
    from . import fem_common as fc
    # When run as script raises:
    #  - `ModuleNotFoundError(ImportError)` (Python 3.6-7), or
    #  - `SystemError` (Python 3.3-5), or
    #  - `ValueError` (Python 2.7).

except (ImportError, SystemError, ValueError):
    import fem_common as fc


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


_DIRECTORY = os.path.dirname(__file__)


try:
    from dolfin import Constant, DirichletBC, Expression

except (ModuleNotFoundError, ImportError):
    logger.warning("Unable to import from dolfin")

else:
    class SpherePointSourcePotentialFEM(fc._SubtractionPointSourcePotentialFEM):
        MAX_ITER = 1000

        def _potential_gradient_normal(self, conductivity=0.0):
            # (1 / r)' = -1 / dst^2
            # -1 * dz / dst
            # dz / dst^3 = dz * (dst^2)^-1.5
            # drx / r * dx / dst = (drx * dx) / (r * dst)
            # = (x * src_x - x * x) / (r * dst)

            dx = '(x[0] - src_x)'
            dy = '(x[1] - src_y)'
            dz = '(x[2] - src_z)'
            drx = 'x[0]'
            dry = 'x[1]'
            drz = 'x[2]'
            dot = f'({dx} * {drx} + {dy} * {dry} + {dz} * {drz})'
            r2 = f'({drx} * {drx} + {dry} * {dry} + {drz} * {drz})'
            dst2 = f'({dx} * {dx} + {dy} * {dy} + {dz} * {dz})'
            return Expression(f'''
                              {0.25 / np.pi} / conductivity
                              * ({dot} / sqrt({dst2} * {r2}) - 1.0)
                              / {dst2}
                              ''',
                              degree=self.degree,
                              domain=self._fm.mesh,
                              src_x=0.0,
                              src_y=0.0,
                              src_z=0.0,
                              conductivity=conductivity)

        def base_conductivity(self, x, y, z):
            return self.config.getfloat('brain', 'conductivity')

        def _add_boundary_conditions(self, x, y, z):
            logger.debug('Defining boundary condition...')
            self._dirichlet_bc = self._boundary_condition(x, y, z)
            logger.debug('Done.  Applying boundary condition to the matrix...')
            self._dirichlet_bc.apply(self._terms_with_unknown)
            logger.debug('Done.  Applying boundary condition to the vector...')
            self._dirichlet_bc.apply(self._known_terms)
            logger.debug('Done.')

        def _boundary_condition(self, x, y, z):
            assert x != 0 or y != 0 or z != 0
            return DirichletBC(self._fm.function_space,
                               Constant(0),
                               "near(x[0], {}) && near(x[1], {}) && near(x[2], {})".format(0, 0, 0),
                               "pointwise")

        def _potential_expression(self, conductivity=0.0):
            dx = '(x[0] - src_x)'
            dy = '(x[1] - src_y)'
            dz = '(x[2] - src_z)'
            drx = 'x[0]'
            dry = 'x[1]'
            drz = 'x[2]'
            r2 = f'({drx} * {drx} + {dry} * {dry} + {drz} * {drz})'
            dst2 = f'({dx} * {dx} + {dy} * {dy} + {dz} * {dz})'
            return Expression(f'''
                              {0.25 / np.pi} / conductivity
                              * (1.0 / sqrt({r2}) - 1.0 / sqrt({dst2}))
                              ''',
                              degree=self.degree,
                              domain=self._fm.mesh,
                              src_x=0.0,
                              src_y=0.0,
                              src_z=0.0,
                              conductivity=conductivity)


    class PointSourceFactoryINI(fc.PointSourceFactoryINI):
        class LazySource(fc.PointSourceFactoryINI.LazySource):
            __slots__ = ()

            def _potential_not_corrected(self, X, Y, Z):
                return self._a * (1. / self._distance(X, Y, Z)
                                  - 1. / self._distance_from_center(X, Y, Z))

            def _distance_from_center(self, X, Y, Z):
                return np.sqrt(np.square(X)
                               + np.square(Y)
                               + np.square(Z))


    if __name__ == '__main__':
        import sys

        logging.basicConfig(level=logging.INFO)

        for config in sys.argv[1:]:
            fem = SpherePointSourcePotentialFEM(config)
            solution_metadata_filename = fem._fm.getpath('fem', 'solution_metadata_filename')
            points = list(fem._fm.functions())
            for i, name in enumerate(points):
                x = fem._fm.getfloat(name, 'x')
                y = fem._fm.getfloat(name, 'y')
                z = fem._fm.getfloat(name, 'z')
                logger.info('{} {:3.1f}%:{}\t(x = {:g}\ty = {:g}\tz = {:g})'.format(
                               config,
                               100. * i / len(points),
                               name,
                               x, y, z))
                filename = fem._fm.getpath(name, 'filename')
                if os.path.exists(filename):
                    logger.info(' found')
                    continue

                logger.info(' solving...')
                function = fem.solve(x, y, z)
                if function is not None:
                    fem._fm.store(name, function,
                                  {'global_preprocessing_time': float(fem.global_preprocessing_time),
                                   'local_preprocessing_time': float(fem.local_preprocessing_time),
                                   'solving_time': float(fem.solving_time),
                                   'base_conductivity': fem.base_conductivity(x, y, z),
                                   })

                    logger.info(' done')
                else:
                    logger.info(' failed')

            fem._fm.write(solution_metadata_filename)
