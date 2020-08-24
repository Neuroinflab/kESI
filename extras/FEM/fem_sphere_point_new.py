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

import configparser
import logging
import os

import numpy as np

try:
    from . import _fem_common as fc
    from . import _fem_common_new as fcn
    # When run as script raises:
    #  - `ModuleNotFoundError(ImportError)` (Python 3.6-7), or
    #  - `SystemError` (Python 3.3-5), or
    #  - `ValueError` (Python 2.7).

except (ImportError, SystemError, ValueError):
    import _fem_common as fc
    import _fem_common_new as fcn


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


_DIRECTORY = os.path.dirname(__file__)


try:
    from dolfin import Constant, DirichletBC, Expression

except (ModuleNotFoundError, ImportError):
    logger.warning("Unable to import from dolfin")

else:
    class SpherePointSourcePotentialFEM(fcn._SubtractionPointSourcePotentialFEM):
        MAX_ITER = 1000

        def _potential_gradient_normal(self, conductivity=0.0):
            # (1 / r)' = -1 / dst^2
            # -1 * dz / dst
            # dz / dst^3 = dz * (dst^2)^-1.5
            # drx / r * dx / dst = (drx * dx) / (r * dst)
            # = (x * src_x - x * x) / (r * dst)

            dx = '(src_x - x[0])'
            dy = '(src_y - x[1])'
            dz = '(src_z - x[2])'
            drx = 'x[0]'
            dry = 'x[1]'
            drz = 'x[2]'
            dot = f'({dx} * {drx} + {dy} * {dry} + {dz} * {drz})'
            r2 = f'({drx} * {drx} + {dry} * {dry} + {drz} * {drz})'
            dst2 = f'({dx} * {dx} + {dy} * {dy} + {dz} * {dz})'
            return Expression(f'''
                              -0.25 / {np.pi} / conductivity
                              * {dot}
                              / sqrt({dst2} * {dst2} * {dst2} * {r2})
                              ''',
                              degree=self.degree,
                              domain=self._fm.mesh,
                              src_x=0.0,
                              src_y=0.0,
                              src_z=0.0,
                              conductivity=conductivity)

        def base_conductivity(self, x, y, z):
            return self.config.getfloat('brain', 'conductivity')

        def _boundary_condition(self, x, y, z):
            assert x != 0 or y != 0 or z != 0
            return DirichletBC(self._fm.function_space,
                               Constant(-self._base_potential_expression(0, 0, 0)),
                               "near(x[0], {}) && near(x[1], {}) && near(x[2], {})".format(0, 0, 0),
                               "pointwise")
