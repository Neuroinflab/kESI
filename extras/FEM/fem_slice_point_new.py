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


try:
    from dolfin import Constant, DirichletBC, Expression

except (ModuleNotFoundError, ImportError):
    logger.warning("Unable to import from dolfin")

else:
    class SlicePointSourcePotentialFEM(fc._SubtractionPointSourcePotentialFEM):
        MAX_ITER = 1000

        def _potential_gradient_normal(self, conductivity=0.0):
            # projection on normal: src_z - x[2]
            # projection on Z axis: x[2] - src_z
            return Expression('''
                              -0.25 / {pi} / conductivity
                              * (src_z - x[2])
                              * pow((src_x - x[0])*(src_x - x[0])
                                    + (src_y - x[1])*(src_y - x[1])
                                    + (src_z - x[2])*(src_z - x[2]),
                                    -1.5)
                              '''.format(pi=np.pi),
                              degree=self.degree,
                              domain=self._fm.mesh,
                              src_x=0.0,
                              src_y=0.0,
                              src_z=0.0,
                              conductivity=conductivity)

        def base_conductivity(self, x, y, z):
            return self.config.getfloat('slice', 'conductivity')

        def _boundary_condition(self, x, y, z):
            approximate_potential = self._potential_expression(self.config.getfloat('saline',
                                                                                    'conductivity'))
            radius = self.config.getfloat('dome', 'radius')
            return DirichletBC(self._fm.function_space,
                               Constant(approximate_potential(0, 0, radius)
                                        - self._base_potential_expression(0, 0, radius)),
                               self._boundaries,
                               self.config.getint('dome', 'surface'))

        def _modify_linear_equation(self, x, y, z):
            logger.debug('Defining boundary condition...')
            self._dirichlet_bc = self._boundary_condition(x, y, z)
            logger.debug('Done.  Applying boundary condition to the matrix...')
            self._dirichlet_bc.apply(self._terms_with_unknown)
            logger.debug('Done.  Applying boundary condition to the vector...')
            self._dirichlet_bc.apply(self._known_terms)
            logger.debug('Done.')

        def _potential_expression(self, conductivity=0.0):
            return Expression('''
                              0.25 / {pi} / conductivity
                              / sqrt((src_x - x[0])*(src_x - x[0])
                                     + (src_y - x[1])*(src_y - x[1])
                                     + (src_z - x[2])*(src_z - x[2]))
                              '''.format(pi=np.pi),
                              degree=self.degree,
                              domain=self._fm.mesh,
                              src_x=0.0,
                              src_y=0.0,
                              src_z=0.0,
                              conductivity=conductivity)


    if __name__ == '__main__':
        import sys

        logging.basicConfig(level=logging.INFO)

        for config in sys.argv[1:]:
            function_manager = fc.FunctionManagerINI(config)
            fem = SlicePointSourcePotentialFEM(function_manager)
            solution_filename_pattern = function_manager.get('fem', 'solution_filename_pattern')
            solution_name_pattern = function_manager.get('fem', 'solution_name_pattern')
            solution_metadata_filename = function_manager.getpath('fem', 'solution_metadata_filename')
            h = fem.config.getfloat('slice', 'thickness')
            k = function_manager.getint('fem', 'k')
            n = 2 ** k + 1
            margin = 0.5 * h / n
            Z = np.linspace(margin, h - margin, n)
            X = np.linspace(0, 0.5 * h - margin, 2 ** (k - 1) + 1)

            for x_idx, x in enumerate(X):
                for y_idx, y in enumerate(X[:x_idx + 1]):
                    logger.info('{} {:3.1f}%\t(x = {:g}\ty = {:g})'.format(config,
                                                                           100 * float(x_idx * (x_idx - 1) // 2 + y_idx) / ((2 ** (k - 1) + 1) * 2 ** (k - 2)),
                                                                           x, y))
                    for z_idx, z in enumerate(Z):
                        name = solution_name_pattern.format(x=x_idx,
                                                            y=y_idx,
                                                            z=z_idx)
                        if function_manager.has_solution(name):
                            filename = function_manager.getpath(name, 'filename')
                            if os.path.exists(filename):
                                logger.info('{} {:3.1f}%\t(x = {:g}\ty = {:g},\tz={:g}) found'.format(
                                    config,
                                    100 * float(x_idx * (x_idx - 1) // 2 + y_idx) / ((2 ** (k - 1) + 1) * 2 ** (k - 2)),
                                    x, y, z))
                                continue

                        logger.info('{} {:3.1f}%\t(x = {:g}\ty = {:g},\tz={:g})'.format(
                            config,
                            100 * float(x_idx * (x_idx - 1) // 2 + y_idx) / ((2 ** (k - 1) + 1) * 2 ** (k - 2)),
                            x, y, z))
                        function = fem.solve(x, y, z)
                        filename = solution_filename_pattern.format(x=x_idx,
                                                                    y=y_idx,
                                                                    z=z_idx)
                        function_manager.store(name, function,
                                      {'filename': filename,
                                       'x': x,
                                       'y': y,
                                       'z': z,
                                       'global_preprocessing_time': float(fem.global_preprocessing_time),
                                       'local_preprocessing_time': float(fem.local_preprocessing_time),
                                       'solving_time': float(fem.solving_time),
                                       'base_conductivity': fem.base_conductivity(x, y, z),
                                       })
                    function_manager.write(solution_metadata_filename)

